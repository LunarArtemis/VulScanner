#include <stdio.h>
#include <ncurses.h>

void clear_area(int startx, int starty, int xsize, int ysize)
{
    int x;

    TRACE_LOG("Clearing area %d,%d / %d,%d\n", startx, starty, xsize, ysize);

    while (ysize > 0)
    {
        x = xsize;
        while (x > 0)
        {
            mvaddch(starty + ysize - 2, startx + x - 2, ' ');
            x--;
        }
        ysize--;
    }
}