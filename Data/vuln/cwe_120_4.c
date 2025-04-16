void vulnerable_memcpy(char *user_input, size_t input_size) {
    char buffer[32];
    memcpy(buffer, user_input, input_size);
}