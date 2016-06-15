#include "Tensor.hh"

#include <iostream>

class GridWorld
{
    enum { PIT, WIN, PLAYER, WALL };

    Tensor<int> world_;
    int height_;
    int width_;

public:
    GridWorld(std::size_t height, std::size_t width) :
        world_({4, height, width}),
        height_(height),
        width_(width)
    {
        // static version
        world_.at({PLAYER, 1,1}) = 1;
        world_.at({PIT, 2,2})    = 1;
        world_.at({WIN, 4,4})    = 1;
        world_.at({WALL, 2,3})   = 1;

        for (std::size_t i=0;i<height;++i)
        {
            world_.at({WALL, i, 0      }) = 1;
            world_.at({WALL, i, width-1}) = 1;
        }
        for (std::size_t i=0;i<width;++i)
        {
            world_.at({WALL, 0       , i}) = 1;
            world_.at({WALL, height-1, i}) = 1;
        }
    }

    void locate(const std::size_t z, std::size_t& y, std::size_t& x)
    {
        for (x = 0; x < TensorUtils<int>::shape(world_)[2]; ++x)
        {
            for (y = 0; y < TensorUtils<int>::shape(world_)[2]; ++y)
            {
                if (world_.at({z,y,x}) == 1)
                    return;
            }
        }
        y = -1;
        x = -1;
    }

    void move(int direction)
    {
        std::size_t cx,cy;
        locate(PLAYER,cy,cx);

        std::size_t x = cx;
        std::size_t y = cy;
        if (direction == 0) x = (x-1)%width_ ;
        if (direction == 1) y = (y-1)%height_;
        if (direction == 2) x = (x+1)%width_ ;
        if (direction == 3) y = (y+1)%height_;

        if (world_.at({WALL,y,x}) == 1) return; // donk..

        world_.at({PLAYER,cy,cx}) = 0;
        world_.at({PLAYER, y, x}) = 1;
    }

    int gameOver()
    {
        std::size_t x,y;
        locate(PLAYER,y,x);

        if (world_.at({PIT,y,x}) == 1) return -1;
        if (world_.at({WIN,y,x}) == 1) return  1;
        return 0;
    }

    void render(std::ostream& os)
    {
        for (std::size_t y = 0; y < height_; ++y)
        {
            for (std::size_t x = 0; x < width_; ++x)
            {
                if      (world_.at({WALL,   y, x})) os << "#";
                else if (world_.at({PIT,    y, x})) os << "-";
                else if (world_.at({WIN,    y, x})) os << "+";
                else if (world_.at({PLAYER, y, x})) os << "P";
                else                                os << " ";
            }
            os << "\n";
        }
    }
};

int main()
{
    // user game
    bool running = true;
    int count = 0; // part of the game.. ??

    GridWorld gworld(6,6);
    while (running)
    {
        int result = gworld.gameOver();
        if (result != 0)
        {
            std::cout << ((result < 0) ? "lost":"win")
                      << " in " << count << " moves\n";
            count = 0;
            gworld = GridWorld(6,6);
        }

        gworld.render(std::cout);

        char move;
        std::cin >> move;
        switch(move)
        {
          case 'a':
              gworld.move(0);
              break;
          case 'w':
              gworld.move(1);
              break;
          case 'd':
              gworld.move(2);
              break;
          case 's':
              gworld.move(3);
              break;
          case 'q':
              running = false;
          default:
              break;
        }
        ++count;
    }

}
