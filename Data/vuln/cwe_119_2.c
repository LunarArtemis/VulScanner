img_filename(const char *mapimgfile, enum imageformat format, char *filename, size_t filename_len)
{
    fc_assert_ret_val(imageformat_is_valid(format), FALSE);

    fc_snprintf(filename, filename_len, "%s.map.%s", mapimgfile, imageformat_name(format));

    return TRUE;
}