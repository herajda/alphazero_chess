#ifndef ROLLING_BOARD_HISTORY_H
#define ROLLING_BOARD_HISTORY_H

#include <deque>
#include <stdexcept> // For std::out_of_range
#include <cstddef>   // For size_t
#include <cassert>   // For assert

#include "chess.hpp" // Include the chess library header

/**
 * @brief Manages a fixed-size rolling history of chess board states.
 *
 * When a new board state is added and the history is full, the oldest
 * state is automatically removed.
 */
class RollingBoardHistory {
public:
    /**
     * @brief Constructs a RollingBoardHistory object.
     * @param max_size The maximum number of board states to keep in history. Must be > 0.
     */
    explicit RollingBoardHistory(size_t max_size) : max_size_(max_size) {
        if (max_size == 0) {
             throw std::invalid_argument("RollingBoardHistory max_size must be greater than 0.");
        }
        // Optional: Reserve space if deque implementation benefits from it
        // history_.reserve(max_size); // deque doesn't have reserve
    }

    /**
     * @brief Adds a copy of the given board state to the history.
     * If the history size exceeds max_size, the oldest state is removed.
     * @param board_state The board state to add.
     */
    void add_state(const chess::Board& board_state) {
        // Check if we need to remove the oldest state first
        if (history_.size() >= max_size_) {
            history_.pop_front();
        }
        // Add a copy of the current state to the back
        history_.push_back(board_state);
    }

    /**
     * @brief Gets the most recently added board state.
     * @return A const reference to the newest board state.
     * @throws std::out_of_range if the history is empty.
     */
    const chess::Board& get_latest_state() const {
        if (empty()) {
            throw std::out_of_range("Cannot get latest state from empty history.");
        }
        return history_.back();
    }

    /**
     * @brief Gets the oldest board state currently in the history.
     * @return A const reference to the oldest board state.
     * @throws std::out_of_range if the history is empty.
     */
    const chess::Board& get_oldest_state() const {
        if (empty()) {
            throw std::out_of_range("Cannot get oldest state from empty history.");
        }
        return history_.front();
    }

    /**
     * @brief Gets the board state at a specific index.
     * Index 0 is the oldest state, index size()-1 is the newest.
     * @param index The index of the state to retrieve.
     * @return A const reference to the board state at the specified index.
     * @throws std::out_of_range if the index is invalid.
     */
    const chess::Board& get_state_at(size_t index) const {
         if (index >= history_.size()) {
             throw std::out_of_range("Index out of range for RollingBoardHistory.");
         }
        // Using operator[] is fine after the bounds check
        return history_[index];
    }

     /**
      * @brief Allows access like an array. Index 0 is oldest, size()-1 is newest.
      * No bounds checking is performed. Use get_state_at() for safe access.
      * @param index The index of the state to retrieve.
      * @return A const reference to the board state at the specified index.
      */
     const chess::Board& operator[](size_t index) const {
        assert(index < history_.size() && "Index out of bounds");
        return history_[index];
     }


    /**
     * @brief Returns the current number of states in the history.
     * @return The number of states stored.
     */
    size_t size() const noexcept {
        return history_.size();
    }

    /**
     * @brief Checks if the history is empty.
     * @return true if the history contains no states, false otherwise.
     */
    bool empty() const noexcept {
        return history_.empty();
    }

    /**
     * @brief Returns the maximum number of states this history can hold.
     * @return The maximum size.
     */
    size_t capacity() const noexcept {
        return max_size_;
    }

    /**
     * @brief Removes all states from the history.
     */
    void clear() noexcept {
        history_.clear();
    }

    // Provide iterators if needed (allow range-based for loops)
    using const_iterator = typename std::deque<chess::Board>::const_iterator;

    const_iterator begin() const noexcept { return history_.begin(); }
    const_iterator cbegin() const noexcept { return history_.cbegin(); }
    const_iterator end() const noexcept { return history_.end(); }
    const_iterator cend() const noexcept { return history_.cend(); }

private:
    std::deque<chess::Board> history_;
    const size_t max_size_;
};

#endif // ROLLING_BOARD_HISTORY_H
