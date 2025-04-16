void pointer_subtraction_heap() {
    char *heap1 = (char *)malloc(10);
    char *heap2 = (char *)malloc(10);
    size_t distance = heap2 - heap1;
    free(heap1);
    free(heap2);
}
