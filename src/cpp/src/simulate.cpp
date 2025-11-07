// src/cpp/src/simulate.cpp
#include "simulate.hpp"
#include "mcts.hpp"
#include "chess_game.hpp"
#include "batch_manager.hpp"

#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <fstream>
#include <vector>
#include <cstdint>
#include <random>
#include <iostream>
#include <memory>
#include <filesystem>
#include <atomic>
#include <csignal> // <-- add this
#include <cctype>
#include <cstring>
#include <array>
#include <string>
#include <string_view>
#include <stdexcept>
#include <unordered_map>
#include <limits>
#include <list>
#include <algorithm>

namespace az73 {

// Global atomic flag for interruption
std::atomic<bool> interrupted{false};

// Signal handler
void handle_sigint(int) {
    interrupted = true;
    std::cerr << "[AZ] Caught SIGINT, stopping simulation..." << std::endl;
}

namespace {

constexpr std::size_t SUP_STATE_FLOATS = 8 * 8 * 119;
constexpr std::size_t SUP_STATE_BYTES  = SUP_STATE_FLOATS * sizeof(float);
constexpr std::size_t SUP_ACTION_BYTES = sizeof(std::int32_t);
constexpr std::size_t SUP_VALUE_BYTES  = sizeof(float);
constexpr std::size_t SUP_RECORD_BYTES = SUP_STATE_BYTES + SUP_ACTION_BYTES + SUP_VALUE_BYTES;
constexpr std::size_t SUP_HEADER_BYTES = sizeof(std::int64_t);
constexpr std::size_t SUP_CHUNK_RECORDS = 1024;
constexpr std::size_t SUP_CHUNK_BYTES   = SUP_RECORD_BYTES * SUP_CHUNK_RECORDS;
constexpr std::size_t SUP_MAX_CACHE_BYTES = 512ull * 1024ull * 1024ull ; // 512 MiB

struct SampleRecord {
    std::array<float, SUP_STATE_FLOATS> state{};
    std::int32_t action{0};
    float value{0.0f};
};

struct ChunkCacheEntry {
    std::shared_ptr<const std::vector<std::uint8_t>> data;
    std::list<std::size_t>::iterator lru_it;
};

struct CachedBufferEntry {
    std::string path;
    std::filesystem::file_time_type timestamp{};
    std::uintmax_t file_size{0};
    std::uintmax_t data_bytes{0};
    std::int64_t total_records{0};
    std::size_t chunk_bytes{0};
    std::size_t max_cached_bytes{0};
    std::size_t cached_bytes{0};

    std::unordered_map<std::size_t, ChunkCacheEntry> chunks;
    std::list<std::size_t> lru_order;
    std::mutex mutex;
};

std::mutex g_supervised_cache_mutex;
std::unordered_map<std::string, std::shared_ptr<CachedBufferEntry>> g_supervised_cache;

std::shared_ptr<CachedBufferEntry> load_supervised_buffer(const std::string &buffer_path,
                                                          std::filesystem::file_time_type timestamp,
                                                          std::uintmax_t file_size) {
    if (file_size < SUP_HEADER_BYTES) {
        throw std::runtime_error("Supervised buffer is too small: " + buffer_path);
    }

    std::int64_t total_records = 0;
    {
        py::gil_scoped_release release;
        std::ifstream input(buffer_path, std::ios::binary);
        if (!input) {
            throw std::runtime_error("Failed to open supervised buffer: " + buffer_path);
        }
        input.read(reinterpret_cast<char*>(&total_records), sizeof(total_records));
        if (input.gcount() != std::streamsize(sizeof(total_records))) {
            throw std::runtime_error("Failed to read supervised buffer header: " + buffer_path);
        }
    }
    if (total_records < 0) {
        throw std::runtime_error("Supervised buffer has negative record count: " + buffer_path);
    }
    const auto data_bytes = static_cast<std::uintmax_t>(total_records) * SUP_RECORD_BYTES;
    const auto expected_size = SUP_HEADER_BYTES + data_bytes;
    if (expected_size > file_size) {
        throw std::runtime_error("Supervised buffer truncated: " + buffer_path);
    }

    const auto capped_data_bytes = std::min<std::uintmax_t>(
        data_bytes,
        static_cast<std::uintmax_t>(std::numeric_limits<std::size_t>::max()));
    const auto preferred_chunk = std::min<std::uintmax_t>(
        static_cast<std::uintmax_t>(SUP_CHUNK_BYTES),
        capped_data_bytes > 0 ? capped_data_bytes : static_cast<std::uintmax_t>(SUP_CHUNK_BYTES));
    const auto chunk_bytes_uint = std::max<std::uintmax_t>(
        static_cast<std::uintmax_t>(SUP_RECORD_BYTES),
        preferred_chunk);
    const auto chunk_bytes = static_cast<std::size_t>(chunk_bytes_uint);

    auto entry = std::make_shared<CachedBufferEntry>();
    entry->path = buffer_path;
    entry->timestamp = timestamp;
    entry->file_size = file_size;
    entry->data_bytes = data_bytes;
    entry->total_records = total_records;
    entry->chunk_bytes = chunk_bytes;
    entry->max_cached_bytes = std::max<std::size_t>(entry->chunk_bytes, SUP_MAX_CACHE_BYTES);
    return entry;
}

std::shared_ptr<CachedBufferEntry> get_supervised_buffer(const std::string &buffer_path) {
    namespace fs = std::filesystem;
    const auto timestamp = fs::last_write_time(buffer_path);
    const auto file_size = fs::file_size(buffer_path);

    {
        std::lock_guard<std::mutex> lock(g_supervised_cache_mutex);
        auto it = g_supervised_cache.find(buffer_path);
        if (it != g_supervised_cache.end() &&
            it->second->timestamp == timestamp &&
            it->second->file_size == file_size) {
            return it->second;
        }
    }

    auto entry = load_supervised_buffer(buffer_path, timestamp, file_size);

    {
        std::lock_guard<std::mutex> lock(g_supervised_cache_mutex);
        auto &slot = g_supervised_cache[buffer_path];
        if (slot && slot->timestamp == timestamp && slot->file_size == file_size) {
            return slot;
        }
        slot = entry;
    }
    return entry;
}

std::mt19937_64 &sampling_rng() {
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    return rng;
}

inline void touch_chunk_locked(CachedBufferEntry &entry,
                               std::unordered_map<std::size_t, ChunkCacheEntry>::iterator it) {
    entry.lru_order.erase(it->second.lru_it);
    entry.lru_order.push_front(it->first);
    it->second.lru_it = entry.lru_order.begin();
}

std::shared_ptr<const std::vector<std::uint8_t>>
acquire_chunk(const std::shared_ptr<CachedBufferEntry> &entry, std::size_t chunk_index) {
    {
        std::lock_guard<std::mutex> lock(entry->mutex);
        auto it = entry->chunks.find(chunk_index);
        if (it != entry->chunks.end()) {
            touch_chunk_locked(*entry, it);
            return it->second.data;
        }
    }

    const auto chunk_offset = static_cast<std::uintmax_t>(chunk_index) * static_cast<std::uintmax_t>(entry->chunk_bytes);
    if (chunk_offset >= entry->data_bytes) {
        throw std::runtime_error("Supervised buffer chunk out of range: " + entry->path);
    }
    const auto bytes_remaining = entry->data_bytes - chunk_offset;
    const auto bytes_to_read_uint = std::min<std::uintmax_t>(bytes_remaining, static_cast<std::uintmax_t>(entry->chunk_bytes));
    if (bytes_to_read_uint > static_cast<std::uintmax_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("Supervised buffer chunk size exceeds platform limit: " + entry->path);
    }
    const auto bytes_to_read = static_cast<std::size_t>(bytes_to_read_uint);

    auto chunk_data = std::make_shared<std::vector<std::uint8_t>>(bytes_to_read);
    std::ifstream input(entry->path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Failed to open supervised buffer: " + entry->path);
    }
    const auto file_offset = static_cast<std::uintmax_t>(SUP_HEADER_BYTES) + chunk_offset;
    if (file_offset > static_cast<std::uintmax_t>(std::numeric_limits<std::streamoff>::max())) {
        throw std::runtime_error("Supervised buffer offset exceeds stream limit: " + entry->path);
    }
    input.seekg(static_cast<std::streamoff>(file_offset), std::ios::beg);
    input.read(reinterpret_cast<char*>(chunk_data->data()), std::streamsize(bytes_to_read));
    if (input.gcount() != std::streamsize(bytes_to_read)) {
        throw std::runtime_error("Failed to read supervised buffer chunk: " + entry->path);
    }

    std::lock_guard<std::mutex> lock(entry->mutex);
    auto it = entry->chunks.find(chunk_index);
    if (it != entry->chunks.end()) {
        touch_chunk_locked(*entry, it);
        return it->second.data;
    }

    entry->lru_order.push_front(chunk_index);
    auto lru_it = entry->lru_order.begin();
    entry->cached_bytes += chunk_data->size();
    entry->chunks.emplace(chunk_index, ChunkCacheEntry{chunk_data, lru_it});

    while (entry->cached_bytes > entry->max_cached_bytes && entry->lru_order.size() > 1) {
        auto victim_idx = entry->lru_order.back();
        entry->lru_order.pop_back();
        auto victim_it = entry->chunks.find(victim_idx);
        if (victim_it != entry->chunks.end()) {
            entry->cached_bytes -= victim_it->second.data->size();
            entry->chunks.erase(victim_it);
        }
    }

    return chunk_data;
}

class SupervisedShuffleBuffer {
public:
    SupervisedShuffleBuffer(std::shared_ptr<CachedBufferEntry> entry,
                            std::size_t requested_capacity,
                            std::int64_t seed_request)
        : m_entry(std::move(entry)),
          m_total_records(resolve_total_records(m_entry)),
          m_capacity(resolve_capacity(requested_capacity, m_total_records)),
          m_warmup_target(std::min(m_capacity, m_total_records)),
          m_seed(resolve_seed(seed_request)),
          m_rng(m_seed),
          m_buffer(m_capacity) {
        if (!m_entry) {
            throw std::runtime_error("Sequential shuffle buffer requires a valid dataset handle.");
        }
        if (m_total_records == 0) {
            throw std::runtime_error("Sequential shuffle buffer received empty dataset: " + m_entry->path);
        }
        if (requested_capacity == 0) {
            throw std::runtime_error("Sequential shuffle buffer requires shuffle_capacity > 0.");
        }
        m_reader = std::thread(&SupervisedShuffleBuffer::reader_main, this);
    }

    ~SupervisedShuffleBuffer() {
        m_stop.store(true, std::memory_order_release);
        m_not_full.notify_all();
        m_not_empty.notify_all();
        if (m_reader.joinable()) {
            m_reader.join();
        }
    }

    SupervisedShuffleBuffer(const SupervisedShuffleBuffer&) = delete;
    SupervisedShuffleBuffer& operator=(const SupervisedShuffleBuffer&) = delete;

    void pop_batch(int batch_size, std::vector<SampleRecord> &out) {
        if (batch_size <= 0) {
            out.clear();
            return;
        }

        out.clear();
        out.reserve(static_cast<std::size_t>(batch_size));

        const std::size_t required = std::min<std::size_t>(m_warmup_target, static_cast<std::size_t>(batch_size));

        std::unique_lock<std::mutex> lock(m_mutex);
        m_not_empty.wait(lock, [&] {
            return m_reader_error || m_stop.load(std::memory_order_acquire) || m_fill_count >= required;
        });

        if (m_reader_error) {
            throw std::runtime_error(m_reader_error_message);
        }

        if (m_fill_count == 0 && m_stop.load(std::memory_order_acquire)) {
            throw std::runtime_error("Sequential shuffle buffer stopped before providing data.");
        }

        for (int i = 0; i < batch_size; ++i) {
            while (m_fill_count == 0 && !m_reader_error && !m_stop.load(std::memory_order_acquire)) {
                m_not_empty.wait(lock);
            }

            if (m_reader_error) {
                throw std::runtime_error(m_reader_error_message);
            }
            if (m_fill_count == 0 && m_stop.load(std::memory_order_acquire)) {
                throw std::runtime_error("Sequential shuffle buffer drained while stopping.");
            }

            std::uniform_int_distribution<std::size_t> dist(0, m_fill_count - 1);
            const std::size_t idx = dist(m_rng);
            out.push_back(m_buffer[idx]);

            --m_fill_count;
            if (idx != m_fill_count) {
                m_buffer[idx] = std::move(m_buffer[m_fill_count]);
            }
        }

        lock.unlock();
        m_not_full.notify_all();
    }

    bool is_compatible(const std::shared_ptr<CachedBufferEntry> &entry,
                       std::size_t requested_capacity,
                       std::int64_t seed_request) const {
        if (!m_entry || !entry) {
            return false;
        }
        if (m_entry->timestamp != entry->timestamp || m_entry->file_size != entry->file_size) {
            return false;
        }
        if (m_capacity != resolve_capacity(requested_capacity, resolve_total_records(entry))) {
            return false;
        }
        if (seed_request >= 0 && m_seed != static_cast<std::uint64_t>(seed_request)) {
            return false;
        }
        return true;
    }

private:
    static std::size_t resolve_total_records(const std::shared_ptr<CachedBufferEntry> &entry) {
        if (!entry) {
            return 0;
        }
        const auto total = entry->total_records;
        if (total <= 0) {
            return 0;
        }
        const auto total_u64 = static_cast<std::uint64_t>(total);
        const auto capped = std::min<std::uint64_t>(total_u64,
            static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()));
        return static_cast<std::size_t>(capped);
    }

    static std::size_t resolve_capacity(std::size_t requested, std::size_t total_records) {
        if (total_records == 0) {
            return 0;
        }
        if (requested == 0) {
            return total_records;
        }
        return std::max<std::size_t>(1, std::min(requested, total_records));
    }

    static std::uint64_t resolve_seed(std::int64_t seed_request) {
        if (seed_request >= 0) {
            return static_cast<std::uint64_t>(seed_request);
        }
        std::random_device rd;
        std::uint64_t s0 = static_cast<std::uint64_t>(rd()) << 32;
        std::uint64_t s1 = static_cast<std::uint64_t>(rd());
        return s0 ^ s1;
    }

    void reader_main() {
        try {
            std::size_t next_record = 0;
            std::size_t current_chunk = std::numeric_limits<std::size_t>::max();
            std::shared_ptr<const std::vector<std::uint8_t>> chunk_data;

            while (!m_stop.load(std::memory_order_acquire)) {
                if (next_record >= m_total_records) {
                    next_record = 0;
                    current_chunk = std::numeric_limits<std::size_t>::max();

                    std::unique_lock<std::mutex> lock(m_mutex);
                    if (m_fill_count > 1) {
                        std::shuffle(m_buffer.begin(), m_buffer.begin() + m_fill_count, m_rng);
                    }
                    ++m_epoch;
                    lock.unlock();
                    continue;
                }

                const std::uintmax_t record_offset_bytes =
                    static_cast<std::uintmax_t>(next_record) * static_cast<std::uintmax_t>(SUP_RECORD_BYTES);
                const std::size_t chunk_index =
                    static_cast<std::size_t>(record_offset_bytes / m_entry->chunk_bytes);
                const std::size_t in_chunk_offset =
                    static_cast<std::size_t>(record_offset_bytes % m_entry->chunk_bytes);

                if (chunk_index != current_chunk) {
                    chunk_data = acquire_chunk(m_entry, chunk_index);
                    current_chunk = chunk_index;
                }

                if (!chunk_data || in_chunk_offset + SUP_RECORD_BYTES > chunk_data->size()) {
                    throw std::runtime_error("Sequential sampler encountered incomplete chunk: " + m_entry->path);
                }

                SampleRecord rec;
                const auto *record_ptr = chunk_data->data() + in_chunk_offset;
                std::memcpy(rec.state.data(), record_ptr, SUP_STATE_BYTES);
                std::memcpy(&rec.action, record_ptr + SUP_STATE_BYTES, SUP_ACTION_BYTES);
                std::memcpy(&rec.value, record_ptr + SUP_STATE_BYTES + SUP_ACTION_BYTES, SUP_VALUE_BYTES);
                ++next_record;

                std::unique_lock<std::mutex> lock(m_mutex);
                m_not_full.wait(lock, [&] {
                    return m_stop.load(std::memory_order_acquire) || m_fill_count < m_capacity;
                });
                if (m_stop.load(std::memory_order_acquire)) {
                    break;
                }
                m_buffer[m_fill_count] = std::move(rec);
                ++m_fill_count;
                lock.unlock();
                m_not_empty.notify_all();
            }
        } catch (const std::exception &ex) {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_reader_error = true;
            m_reader_error_message = ex.what();
            lock.unlock();
            m_not_empty.notify_all();
        }
    }

    std::shared_ptr<CachedBufferEntry> m_entry;
    const std::size_t m_total_records;
    const std::size_t m_capacity;
    const std::size_t m_warmup_target;
    const std::uint64_t m_seed;

    std::mt19937_64 m_rng;
    std::vector<SampleRecord> m_buffer;

    std::mutex m_mutex;
    std::condition_variable m_not_empty;
    std::condition_variable m_not_full;

    std::size_t m_fill_count{0};
    std::size_t m_epoch{0};

    std::atomic<bool> m_stop{false};
    bool m_reader_error{false};
    std::string m_reader_error_message;

    std::thread m_reader;
};

struct ShufflePipelineSlot {
    std::shared_ptr<SupervisedShuffleBuffer> pipeline;
};

std::mutex g_shuffle_mutex;
std::unordered_map<std::string, ShufflePipelineSlot> g_shuffle_pipelines;

std::shared_ptr<SupervisedShuffleBuffer>
get_shuffle_pipeline(const std::string &buffer_path,
                     const std::shared_ptr<CachedBufferEntry> &entry,
                     std::size_t requested_capacity,
                     std::int64_t seed_request) {
    {
        std::lock_guard<std::mutex> lock(g_shuffle_mutex);
        auto it = g_shuffle_pipelines.find(buffer_path);
        if (it != g_shuffle_pipelines.end() && it->second.pipeline &&
            it->second.pipeline->is_compatible(entry, requested_capacity, seed_request)) {
            return it->second.pipeline;
        }
    }

    auto new_pipeline = std::make_shared<SupervisedShuffleBuffer>(entry, requested_capacity, seed_request);
    std::shared_ptr<SupervisedShuffleBuffer> old_pipeline;
    std::shared_ptr<SupervisedShuffleBuffer> result;

    {
        std::lock_guard<std::mutex> lock(g_shuffle_mutex);
        auto &slot = g_shuffle_pipelines[buffer_path];
        if (slot.pipeline &&
            slot.pipeline->is_compatible(entry, requested_capacity, seed_request)) {
            result = slot.pipeline;
        } else {
            old_pipeline = std::move(slot.pipeline);
            slot.pipeline = new_pipeline;
            result = slot.pipeline;
        }
    }
    // ensure old pipeline is destroyed outside the lock
    old_pipeline.reset();
    return result;
}

struct LimitReached : std::exception {
    const char* what() const noexcept override { return "PGN limit reached"; }
};

inline std::string trim_copy(std::string_view sv) {
    std::size_t begin = 0;
    std::size_t end = sv.size();
    while (begin < end && std::isspace(static_cast<unsigned char>(sv[begin]))) {
        ++begin;
    }
    while (end > begin && std::isspace(static_cast<unsigned char>(sv[end - 1]))) {
        --end;
    }
    return std::string(sv.substr(begin, end - begin));
}

inline std::string normalise_result(std::string_view sv) {
    std::string res;
    res.reserve(sv.size());
    for (std::size_t i = 0; i < sv.size(); ++i) {
        unsigned char c = static_cast<unsigned char>(sv[i]);
        if (std::isspace(c)) continue;
        if (c == '*') {
            return "*";
        }
        // Handle UTF-8 en dash (0xE2 0x80 0x93)
        if (c == 0xE2 && i + 2 < sv.size()) {
            unsigned char b1 = static_cast<unsigned char>(sv[i + 1]);
            unsigned char b2 = static_cast<unsigned char>(sv[i + 2]);
            if (b1 == 0x80 && b2 == 0x93) {
                res.push_back('-');
                i += 2;
                continue;
            }
        }
        // Handle UTF-8 one-half (½) => "1/2"
        if (c == 0xC2 && i + 1 < sv.size()) {
            unsigned char b1 = static_cast<unsigned char>(sv[i + 1]);
            if (b1 == 0xBD) {
                res.append("1/2");
                ++i;
                continue;
            }
        }
        res.push_back(static_cast<char>(c));
    }
    return res;
}

inline int result_to_code(const std::string &res) {
    if (res == "1-0") return 1;
    if (res == "0-1") return 0;
    if (res == "1/2-1/2" || res == "1/2") return -1;
    return 2; // unknown
}

inline float orient_value(int winner_code, int to_play) {
    if (winner_code == -1) return 0.0f;
    return (to_play == winner_code) ? 1.0f : -1.0f;
}

struct SupervisedVisitor : chess::pgn::Visitor {
    std::ofstream &out;
    std::int64_t &record_count;
    std::int64_t &processed_games;
    const std::int64_t max_games;

    ChessGame game;
    std::vector<ChessGame::Tensor> states;
    std::vector<std::int32_t> actions;
    std::vector<int> players;
    std::string result_tag{"*"};
    bool skip_current{false};

    SupervisedVisitor(std::ofstream &out_stream,
                      std::int64_t &records,
                      std::int64_t &games,
                      std::int64_t max_games_limit)
        : out(out_stream),
          record_count(records),
          processed_games(games),
          max_games(max_games_limit) {
        states.reserve(128);
        actions.reserve(128);
        players.reserve(128);
    }

    void startPgn() override {
        if (max_games >= 0 && processed_games >= max_games) {
            throw LimitReached{};
        }
        skip_current = false;
        result_tag = "*";
        game = ChessGame();
        states.clear();
        actions.clear();
        players.clear();
    }

    void header(std::string_view key, std::string_view value) override {
        if (skip_current) return;
        if (key == "Result") {
            result_tag = normalise_result(value);
        }
    }

    void startMoves() override {
        // nothing to do here
    }

    void move(std::string_view san, std::string_view /*comment*/) override {
        if (skip_current) return;
        const std::string move_str = trim_copy(san);
        if (move_str.empty()) {
            skip_current = true;
            states.clear();
            actions.clear();
            players.clear();
            return;
        }
        if (move_str == "1-0" || move_str == "0-1" || move_str == "1/2-1/2" || move_str == "1/2" || move_str == "*") {
            return;
        }
        try {
            const chess::Board board = game.currentBoard();
            chess::Move mv = chess::uci::parseSan(board, move_str);
            if (mv == chess::Move::NO_MOVE) {
                skip_current = true;
                states.clear();
                actions.clear();
                players.clear();
                return;
            }
            const std::uint16_t encoded = az73::encode(board, mv);
            if (encoded >= ACTION_SPACE) {
                skip_current = true;
                states.clear();
                actions.clear();
                players.clear();
                return;
            }
            states.push_back(game.encodeTensor());
            actions.push_back(static_cast<std::int32_t>(encoded));
            players.push_back(game.to_play());
            game.makeMove(mv);
        } catch (const std::exception &) {
            skip_current = true;
            states.clear();
            actions.clear();
            players.clear();
        }
    }

    void endPgn() override {
        if (!skip_current) {
            const int winner = result_to_code(result_tag);
            if (winner != 2) {
                for (std::size_t i = 0; i < states.size(); ++i) {
                    const auto &tensor = states[i];
                    out.write(reinterpret_cast<const char*>(tensor.data()), SUP_STATE_BYTES);
                    const std::int32_t action = actions[i];
                    out.write(reinterpret_cast<const char*>(&action), sizeof(action));
                    const float value = orient_value(winner, players[i]);
                    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
                    ++record_count;
                }
            }
        }
        ++processed_games;
    }
};

} // anonymous namespace

// -- PGN → supervised dataset -------------------------------------------------
void convert_pgn_to_supervised_buffer(const std::string &pgn_path,
                                      const std::string &output_path,
                                      std::int64_t max_games) {
    std::ifstream input(pgn_path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Failed to open PGN file: " + pgn_path);
    }

    std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("Failed to open output file: " + output_path);
    }

    std::int64_t record_count = 0;
    std::int64_t games_seen = 0;

    // reserve header slot
    output.write(reinterpret_cast<const char*>(&record_count), sizeof(record_count));

    SupervisedVisitor visitor(output, record_count, games_seen, max_games);

    chess::pgn::StreamParser parser(input);
    try {
        py::gil_scoped_release release;
        auto err = parser.readGames(visitor);
        if (err.hasError()) {
            throw std::runtime_error("PGN parse error: " + err.message());
        }
    } catch (const LimitReached &) {
        // reached the requested number of games, this is expected
    }

    output.seekp(0);
    output.write(reinterpret_cast<const char*>(&record_count), sizeof(record_count));
    output.flush();
}

py::list sample_supervised_batch(const std::string &buffer_path, int batch_size) {
    if (batch_size <= 0) {
        return py::list();
    }

    auto buffer = get_supervised_buffer(buffer_path);
    if (!buffer || buffer->total_records <= 0) {
        return py::list();
    }

    if (buffer->chunk_bytes == 0) {
        throw std::runtime_error("Supervised buffer chunk size is zero: " + buffer_path);
    }

    const auto total_records = buffer->total_records;
    std::vector<SampleRecord> raw;
    raw.reserve(batch_size);

    {
        py::gil_scoped_release release;
        auto &rng = sampling_rng();
        std::uniform_int_distribution<std::int64_t> dist(0, total_records - 1);

        for (int i = 0; i < batch_size; ++i) {
            const std::int64_t idx = dist(rng);
            const auto record_offset = static_cast<std::uintmax_t>(idx) * static_cast<std::uintmax_t>(SUP_RECORD_BYTES);
            if (record_offset >= buffer->data_bytes) {
                throw std::runtime_error("Supervised buffer record out of range: " + buffer_path);
            }
            const auto chunk_index = static_cast<std::size_t>(record_offset / buffer->chunk_bytes);
            const auto in_chunk_offset_uint = record_offset % buffer->chunk_bytes;
            if (in_chunk_offset_uint > static_cast<std::uintmax_t>(std::numeric_limits<std::size_t>::max())) {
                throw std::runtime_error("Supervised buffer offset overflow: " + buffer_path);
            }
            const auto in_chunk_offset = static_cast<std::size_t>(in_chunk_offset_uint);

            auto chunk_data = acquire_chunk(buffer, chunk_index);
            if (!chunk_data || in_chunk_offset + SUP_RECORD_BYTES > chunk_data->size()) {
                throw std::runtime_error("Supervised buffer chunk insufficient data: " + buffer_path);
            }
            const auto *record_ptr = chunk_data->data() + in_chunk_offset;

            SampleRecord rec;
            std::memcpy(rec.state.data(), record_ptr, SUP_STATE_BYTES);
            std::memcpy(&rec.action, record_ptr + SUP_STATE_BYTES, SUP_ACTION_BYTES);
            std::memcpy(&rec.value, record_ptr + SUP_STATE_BYTES + SUP_ACTION_BYTES, SUP_VALUE_BYTES);
            raw.push_back(std::move(rec));
        }
    }

    py::list batch;
    for (auto &rec : raw) {
        py::array_t<float> state({8, 8, 119});
        std::memcpy(state.mutable_data(), rec.state.data(), SUP_STATE_BYTES);
        batch.append(py::make_tuple(state, rec.action, rec.value));
    }
    return batch;
}

py::list sample_supervised_batch_v2(const std::string &buffer_path,
                                    int batch_size,
                                    std::size_t shuffle_capacity,
                                    std::int64_t seed) {
    if (batch_size <= 0) {
        return py::list();
    }
    if (shuffle_capacity == 0) {
        throw std::runtime_error("sample_supervised_batch_v2 requires shuffle_capacity > 0.");
    }

    auto buffer = get_supervised_buffer(buffer_path);
    if (!buffer || buffer->total_records <= 0) {
        return py::list();
    }

    auto pipeline = get_shuffle_pipeline(buffer_path, buffer, shuffle_capacity, seed);
    std::vector<SampleRecord> raw;
    {
        py::gil_scoped_release release;
        pipeline->pop_batch(batch_size, raw);
    }

    py::list batch;
    for (auto &rec : raw) {
        py::array_t<float> state({8, 8, 119});
        std::memcpy(state.mutable_data(), rec.state.data(), SUP_STATE_BYTES);
        batch.append(py::make_tuple(state, rec.action, rec.value));
    }
    return batch;
}

std::int64_t supervised_buffer_size(const std::string &buffer_path) {
    std::ifstream input(buffer_path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Failed to open supervised buffer: " + buffer_path);
    }
    std::int64_t total_records = 0;
    input.read(reinterpret_cast<char*>(&total_records), sizeof(total_records));
    return total_records;
}

// --------------------------------------------------------------------
// Evaluate vs STOCKFISH (UCI) -- threaded & batched
// --------------------------------------------------------------------
namespace {

// very small helper: spawn Stockfish, return a pair of i/o streams
struct UciEngine {
    std::string bin;
    FILE *in  = nullptr;   // write TO engine
    FILE *out = nullptr;   // read  FROM engine

    explicit UciEngine(const std::string &path) : bin(path) {
        int in_pipe [2], out_pipe[2];
        if (pipe(in_pipe) == -1 || pipe(out_pipe) == -1) {
            throw std::runtime_error("Failed to create pipes for Stockfish process.");
        }

        pid_t pid = fork();
        if (pid == 0) {               // child → Stockfish
            dup2(in_pipe [0], STDIN_FILENO);
            dup2(out_pipe[1], STDOUT_FILENO);
            close(in_pipe [1]); close(out_pipe[0]);
            execl(bin.c_str(), bin.c_str(), nullptr);
            _exit(127);
        }
        // parent
        close(in_pipe [0]);  close(out_pipe[1]);
        in  = fdopen(in_pipe [1],  "w");
        out = fdopen(out_pipe[0],  "r");

        // initialise UCI
        fprintf(in, "uci\n");  fflush(in);
        wait_ready();
    }
    ~UciEngine() { if (in) { fputs("quit\n", in); fflush(in);} }

    void send(const std::string &cmd) { fputs(cmd.c_str(), in); fputc('\n', in); fflush(in);}
    std::string readline() {
        char buf[256];
        if (!fgets(buf, sizeof(buf), out)) return {};
        return std::string(buf);
    }
    void wait_ready() {
        send("isready");
        std::string s;
        while ((s = readline()).find("readyok") == std::string::npos) {}
    }
    void set_options(int elo, int threads) {
        if (elo >= 0) {
            send("setoption name UCI_LimitStrength value true");
            send("setoption name UCI_Elo          value " + std::to_string(elo));
        }
        send("setoption name Threads value " + std::to_string(threads));
        wait_ready();
    }

    // get bestmove in UCI for the current position
    std::string bestmove(const std::string &fen,
                         int depth, bool use_depth) {
        send("position fen " + fen);
        if (use_depth) send("go depth " + std::to_string(depth));
        else           send("go movetime 100");          // fallback
        std::string s;
        while ((s = readline()).rfind("bestmove ",0)) {}  // wait
        auto mv   = s.substr(9);           // drop "bestmove "
        mv.erase(mv.find_first_of(" \t\r\n")); // trim right
        return mv;
    }
}; // struct UciEngine
}  // anonymous namespace

    // helper: returns a *legal* chess::Move that corresponds to action_id
    static chess::Move id_to_legal_move(const chess::Board &board,
                                        uint16_t action_id)
    {
        using namespace chess;
        Movelist legal;
        movegen::legalmoves<movegen::MoveGenType::ALL>(legal, board,
                PieceGenType::PAWN   | PieceGenType::KNIGHT | PieceGenType::BISHOP |
                PieceGenType::ROOK   | PieceGenType::QUEEN  | PieceGenType::KING);
        
        for (const Move &m : legal)           // round-trip test
            if (encode(board, m) == action_id)
                return m;                     // found exact match
        
        // *** should never happen, but be defensive ***
        return legal.empty() ? Move::NULL_MOVE : legal[0];
    }
// public API
std::tuple<int,int,int,int,int,int>
evaluate_vs_stockfish(
    const std::string &model_path,
    const std::string &sf_bin,
    int games_per_color,
    int num_threads,
    int sf_depth,
    int sf_elo,
    int num_simulations,
    double alpha,
    double epsilon,
    int sampling_moves)
{
    using namespace chess;
    constexpr std::size_t GPU_BATCH = 256;
    BatchManager::instance().init(model_path, GPU_BATCH);

    std::atomic<int> wW{0}, wL{0}, wD{0}, bW{0}, bL{0}, bD{0};
    std::atomic<int> gidx{0};

    auto worker = [&](int tid) {
        UciEngine sf(sf_bin);
        sf.set_options(sf_elo, 1);   // 1 thread/engine

        while (true) {
            int idx = gidx.fetch_add(1);
            if (idx >= 2 * games_per_color) break;

            bool agent_is_white = (idx < games_per_color);
            int  us_turn_code  = agent_is_white ? 1 : 0;

            ChessGame game;
            MCTArgs args{num_simulations, alpha, epsilon, sampling_moves};

            while (!game.winner().has_value()) {
                if (game.to_play() == us_turn_code) {
                    auto policy = run_mcts(game, args);
                    auto legal  = game.legalMoves();
                    uint16_t best = *std::max_element(
                        legal.begin(), legal.end(),
                        [&](uint16_t a, uint16_t b){return policy[a] < policy[b];});
                    game.makeMove(best);

                } else {
                    std::string fen   = game.currentBoard().getFen();

                    std::string uci   = sf.bestmove(fen, sf_depth, sf_elo < 0);
                    chess::Move  mv   = chess::uci::uciToMove(game.currentBoard(), uci);
                    uint16_t a = encode(game.currentBoard(), mv);

                    game.makeMove(a);
                }
            }
            int res = *game.winner();       // 1 white win, 0 black win, -1 draw
            if (agent_is_white) {
                if (res ==  1) wW++; else
                if (res ==  0) wL++; else wD++;
            } else {
                if (res ==  0) bW++; else
                if (res ==  1) bL++; else bD++;
            }
        }
    };

    std::vector<std::thread> pool;
    for (int t = 0; t < num_threads; ++t) pool.emplace_back(worker, t);
    for (auto &th : pool) th.join();

    return {wW.load(), wL.load(), wD.load(),
            bW.load(), bL.load(), bD.load()};
}


// --------------------------------------------------------------------
// Evaluate vs random
// --------------------------------------------------------------------
std::tuple<int,int,int,int,int,int>
evaluate_vs_random(
    const std::string &model_path,
    int num_games_per_color,
    int num_threads,
    int num_simulations,
    double alpha,
    double epsilon,
    int sampling_moves
) {
    // 1) initialize model (batch size == num_threads)
    constexpr std::size_t GPU_BATCH = 256;
    BatchManager::instance().init(model_path, GPU_BATCH);
    py::gil_scoped_release no_gil;

    // 2) counters
    std::atomic<int> w_win{0}, w_loss{0}, w_draw{0};
    std::atomic<int> b_win{0}, b_loss{0}, b_draw{0};

    // 3) shared atomic index for “which game to play next”
    //    we want total = 2 * num_games_per_color (first half = White‐side games, next half = Black‐side)
    std::atomic<int> game_idx{0};

    auto worker = [&](int tid) {
        std::mt19937_64 rng{std::random_device{}()};

        while (true) {
            int idx = game_idx.fetch_add(1, std::memory_order_relaxed);
            if (idx >= 2 * num_games_per_color)
                break;

            // determine color from idx:
            //   idx in [0 .. num_games_per_color-1]  → play as White
            //   idx in [num_games_per_color .. 2*num_games_per_color-1] → play as Black
            int color = (idx < num_games_per_color ? 1 : 0);
            ChessGame game;
            MCTArgs args{num_simulations, alpha, epsilon, sampling_moves};

            // play until game over
            while (!game.winner().has_value()) {
                if (game.to_play() == color) {
                    // our agent turn
                    auto policy = run_mcts(game, args);
                    auto legal = game.legalMoves();
                    uint16_t best = legal.front();
                    float best_p = policy[best];
                    for (auto a : legal) {
                        if (policy[a] > best_p) {
                            best_p = policy[a];
                            best = a;
                        }
                    }
                    game.makeMove(best);

                } else {
                    // random‐move baseline
                    auto legal = game.legalMoves();
                    std::uniform_int_distribution<size_t> uni(0, legal.size() - 1);
                    size_t random_idx = uni(rng);
                    game.makeMove(legal[random_idx]);
                }
            }

            int outcome = *game.winner();  // 1=White, 0=Black, –1=draw
            if (color == 1) {
                // we played White
                if      (outcome == -1) w_draw++;
                else if (outcome ==  1) w_win++;
                else                    w_loss++;
            } else {
                // we played Black
                if      (outcome == -1) b_draw++;
                else if (outcome ==  0) b_win++;
                else                    b_loss++;
            }
        }
    };

    // 4) launch exactly num_threads threads
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back(worker, t);
    }
    for (auto &th : threads) th.join();

    // 5) return aggregated counts
    return {
        w_win.load(),  w_loss.load(),  w_draw.load(),
        b_win.load(),  b_loss.load(),  b_draw.load()
    };
}

// --------------------------------------------------------------------
// Number of floats per record: state (8x8x119) + policy (8x8x73) + z
static constexpr uint64_t RECORD_FLOATS = 8ULL * 8 * 119 + 8ULL * 8 * 73 + 1;
static constexpr uint64_t RECORD_BYTES = RECORD_FLOATS * sizeof(float);
static constexpr uint64_t HEADER_BYTES = 24; // [capacity(8), size(8), head(8)]

// Helpers to read/write 64-bit header fields
static int64_t read_i64(std::fstream &f, std::streamoff off) {
    int64_t v;
    f.seekg(off);
    f.read(reinterpret_cast<char*>(&v), sizeof(v));
    return v;
}

static void write_i64(std::fstream &f, std::streamoff off, int64_t v) {
    f.seekp(off);
    f.write(reinterpret_cast<const char*>(&v), sizeof(v));
    f.flush();
}

void simulate_games_buffered(
    const std::string &model_path,
    int num_games,
    int num_threads,
    int num_simulations,
    double alpha,
    double epsilon,
    int sampling_moves,
    const std::string &filename,
    int64_t capacity
) {
    // Install signal handler (only once, safe for repeated calls)
    static std::once_flag sig_flag;
    std::call_once(sig_flag, []() {
        std::signal(SIGINT, handle_sigint);
    });



    // 1) Load model
    BatchManager::instance().init(model_path, 64);

    py::gil_scoped_release no_gil;

    // 2) Open (or create) ring-buffer file
    std::fstream f(filename, std::ios::in | std::ios::out | std::ios::binary);
    if (!f) {
        std::cerr << "[AZ] creating new buffer file\n";
        std::ofstream of(filename, std::ios::binary | std::ios::trunc);
        int64_t zero = 0;
        of.write(reinterpret_cast<const char*>(&capacity), sizeof(capacity));
        of.write(reinterpret_cast<const char*>(&zero), sizeof(zero));
        of.write(reinterpret_cast<const char*>(&zero), sizeof(zero));
        of.close();
        std::filesystem::resize_file(filename, HEADER_BYTES + capacity * RECORD_BYTES);
        f.open(filename, std::ios::in | std::ios::out | std::ios::binary);
    }
    std::cerr << "[AZ] buffer file opened\n";

    // 3) Read header
    int64_t size = read_i64(f, 8);
    int64_t head = read_i64(f, 16);
    std::cerr << "[AZ] initial size=" << size << " head=" << head << "\n";

    std::mutex file_mtx;
    auto append_one = [&](const std::vector<float> &state, const std::vector<float> &policy, float z) {
        std::lock_guard lk(file_mtx);
        uint64_t idx = head % capacity;
        uint64_t offset = HEADER_BYTES + idx * RECORD_BYTES;
        f.seekp(offset);
        f.write(reinterpret_cast<const char*>(state.data()), state.size() * sizeof(float));
        f.write(reinterpret_cast<const char*>(policy.data()), policy.size() * sizeof(float));
        f.write(reinterpret_cast<const char*>(&z), sizeof(z));
        f.flush();
        head = (head + 1) % capacity;
        if (size < capacity) ++size;
        write_i64(f, 8, size);
        write_i64(f, 16, head);
        std::cerr << "[AZ] new size=" << size << " new head=" << head << "\n";
    };

    // --- dynamic, perfectly-balanced scheduling --------------------
    std::atomic<int> game_idx{0};
    std::vector<std::thread> threads;
    threads.reserve(num_threads);


    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            // each thread gets its own RNG
            std::mt19937_64 rng{std::random_device{}()};

            // grab one game at a time
            while (true) {
                // Check for interruption before starting a new game
                if (interrupted) break;

                int idx = game_idx.fetch_add(1, std::memory_order_relaxed);
                if (idx >= num_games)
                    break;

                // ----- simulate exactly one game -----
                // (copy the body of 'worker' here, but for a single game)
                std::vector<std::vector<float>> states;
                std::vector<std::vector<float>> policies;
                std::vector<float> toplays;

                ChessGame game;
                MCTArgs args{ num_simulations, alpha, epsilon, sampling_moves };
                while (!game.winner().has_value()) {
                    // Check for interruption inside game loop
                    if (interrupted) break;
                    auto raw = game.encodeTensor();

                    std::vector<float> flat;
                    flat.reserve(raw.size());
                    for (uint16_t b : raw)
                        flat.push_back(static_cast<float>(b));      // 0.f / 1.f

                    auto policy = run_mcts(game, args);
                    //std::cout << "[AZ] policy size = " << policy.size() << std::endl;
                    auto legal = game.legalMoves();

                    std::vector<float> masked(ACTION_SPACE, 0.0f);
                    for (auto mv : legal) masked[mv] = policy[mv];
                    float sum = std::accumulate(masked.begin(), masked.end(), 0.0f);

                    int action;
                    if (sum == 0.0f) {
                        std::uniform_int_distribution<size_t> u(0, legal.size() - 1);
                        action = legal[u(rng)];
                    } else if ((int)states.size() >= sampling_moves) {
                        action = int(std::distance(masked.begin(),
                                                   std::max_element(masked.begin(), masked.end())));
                    } else {
                        std::discrete_distribution<int> d(masked.begin(), masked.end());
                        action = d(rng);
                    }

                    // Print FEN and action in UCI format
                    //std::cout << "[AZ] size " << size << " FEN: " << game.currentBoard().getFen() << std::endl;
                    //std::cout << "[AZ] Action: " 
                    //          << chess::uci::moveToUci(az73::decode_action(action, game.currentBoard()))
                    //          << std::endl;

                    states.push_back(std::move(flat));
                    policies.push_back(std::move(policy));
                    toplays.push_back(float(game.to_play()));
                    game.makeMove(uint16_t(action));
                }

                // If interrupted, don't write partial games
                if (interrupted) break;

                int w = *game.winner();
                for (size_t k = 0; k < states.size(); ++k) {
                    float z = (w == -1 ? 0.0f
                                : toplays[k] == float(w) ? 1.0f
                                                         : -1.0f);
                    append_one(states[k], policies[k], z);
                }
                // ----------------------------------------
            }
        });
    }
    for (auto &th : threads) th.join();
    f.close();
    std::cerr << "[AZ] simulate_games_buffered DONE\n";

    // If interrupted, raise Python KeyboardInterrupt
    if (interrupted) {
        PyErr_SetInterrupt(); // sets the Python interrupt flag
        throw py::error_already_set();
    }
}

} // namespace az73
namespace py = pybind11;
using az73::BatchManager;
using az73::MCTArgs;
using az73::run_mcts;

PYBIND11_MODULE(chess_engine, m) {
    m.doc() = "C++ self-play with batched inference and on-disk ring buffer";
    m.def("convert_pgn_to_supervised_buffer", &az73::convert_pgn_to_supervised_buffer,
          py::arg("pgn_path"),
          py::arg("output_path"),
          py::arg("max_games") = -1,
          "Convert a PGN file to a binary supervised training buffer using the chess-library parser.");
    m.def("sample_supervised_batch", &az73::sample_supervised_batch,
          py::arg("buffer_path"),
          py::arg("batch_size"),
          "Sample a batch of (state, action, value) tuples from a supervised buffer.");
    m.def("sample_supervised_batch_v2", &az73::sample_supervised_batch_v2,
          py::arg("buffer_path"),
          py::arg("batch_size"),
          py::arg("shuffle_buffer_size") = static_cast<std::size_t>(1'000'000),
          py::arg("seed") = -1,
          "Stream batches through a sequential reader with a shuffle buffer to reduce random I/O.");
    m.def("supervised_buffer_size", &az73::supervised_buffer_size,
          py::arg("buffer_path"),
          "Return the number of supervised positions stored in a buffer.");
    m.def("simulate_games_buffered", &az73::simulate_games_buffered,
        py::arg("model_path"),
        py::arg("num_games"),
        py::arg("num_threads"),
        py::arg("num_simulations"),
        py::arg("alpha"),
        py::arg("epsilon"),
        py::arg("sampling_moves"),
        py::arg("filename"),
        py::arg("replay_buffer_capacity"));
    
    m.def("evaluate_vs_random", &az73::evaluate_vs_random,
          py::arg("model_path"),
          py::arg("games_per_color"),
          py::arg("num_threads"),
          py::arg("num_simulations"),
          py::arg("alpha"),
          py::arg("epsilon"),
          py::arg("sampling_moves"),
          "Evaluate the AlphaZero agent vs a random agent, returning "
          "(white_wins, white_losses, white_draws, black_wins, black_losses, black_draws).");
    m.def("select_move",
        // Lambda takes: path to TorchScript, FEN, MCTS params → returns best flat action
        [](const std::string &model_path,
           const std::string &fen,
           int num_simulations,
           double alpha,
           double epsilon,
           int sampling_moves) -> uint16_t
        {
            // 1) Initialize the batched model (use a reasonable batch size)
            constexpr size_t BATCH = 1;
            BatchManager::instance().init(model_path, BATCH);

            // 2) Build game from FEN
            ChessGame game(fen);

            // 3) Run MCTS
            MCTArgs args{num_simulations, alpha, epsilon, sampling_moves};
            auto policy = run_mcts(game, args);

            // 4) Pick the highest-probability legal move
            auto legal = game.legalMoves();
            uint16_t best = legal.front();
            float best_p = policy[best];
            for (auto a : legal) {
                if (policy[a] > best_p) {
                    best_p = policy[a];
                    best   = a;
                }
            }
            return best;
        },
        py::arg("model_path"),
        py::arg("fen"),
        py::arg("num_simulations"),
        py::arg("alpha"),
        py::arg("epsilon"),
        py::arg("sampling_moves"),
        R"pbdoc(
            select_move(model_path, fen, num_simulations, alpha, epsilon, sampling_moves) -> action_index

            Run MCTS from the given FEN and return the chosen move (flattened 0–4671).
        )pbdoc"
    ); 
    m.def("evaluate_vs_stockfish", &az73::evaluate_vs_stockfish,
      py::arg("model_path"),
      py::arg("sf_bin"),
      py::arg("games_per_color"),
      py::arg("num_threads"),
      py::arg("sf_depth"),
      py::arg("sf_elo"),
      py::arg("num_simulations"),
      py::arg("alpha"),
      py::arg("epsilon"),
      py::arg("sampling_moves"),
      "Evaluate the AlphaZero agent against Stockfish and return "
      "(wW,wL,wD,bW,bL,bD).");
}
