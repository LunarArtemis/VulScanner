void pointer_subtraction_negative() {
    char buffer[10];
    char *start = &buffer[7];
    char *end = &buffer[3];
    size_t size = end - start;
}
