using System;
using System.Drawing;

namespace ConsoleApp1
{
    class Program
    {
        static void Main(string[] args)
        {
            Bitmap[] b = new Bitmap[9];
            int img;
            for (int i = 0; i < 9; i++)
            {
                img = 600 + i * 50;
                b[i] = new Bitmap("C:\\Users\\user\\Desktop\\Abhilash\\Imp\\CEERI\\NN\\" + img + "nm.png");
                Console.WriteLine(b[i].Width + "--" + b[i].Height);
            }
            Console.ReadKey();

     //       Color color = b1.GetPixel();
        }
    }
}
