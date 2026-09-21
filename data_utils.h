#ifndef DATA_UTILS_H
#define DATA_UTILS_H

#include <cerrno>
#include <fcntl.h>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

inline bool ReadFile(const std::string &path, size_t expectedSize, void *buffer, size_t bufferSize)
{
    if (buffer == nullptr || expectedSize > bufferSize) {
        return false;
    }
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    file.seekg(0, std::ios::end);
    const auto end = file.tellg();
    if (end < 0 || static_cast<size_t>(end) != expectedSize) {
        return false;
    }
    file.seekg(0, std::ios::beg);
    file.read(static_cast<char *>(buffer), expectedSize);
    return static_cast<bool>(file);
}

inline bool WriteFile(const std::string &path, const void *buffer, size_t size)
{
    if (buffer == nullptr && size != 0) {
        return false;
    }
    const int fd = open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
    if (fd < 0) {
        return false;
    }
    size_t written = 0;
    while (written < size) {
        const ssize_t count = write(fd, static_cast<const char *>(buffer) + written, size - written);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            close(fd);
            return false;
        }
        written += static_cast<size_t>(count);
    }
    return close(fd) == 0;
}

#endif
