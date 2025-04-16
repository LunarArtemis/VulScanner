void pointer_subtraction_out_of_bounds() {
    char data[8];
    char *p1 = &data[8];
    char *p2 = &data[2];
    size_t len = p1 - p2;
}
