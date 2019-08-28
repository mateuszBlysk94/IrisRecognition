/*
 *  1. Pętla - wczytanie po kolei wszystkich zdjęć ze wskazanego folderu, zapisanie wynikowego zdjęcia.
 *  2. Pobranie zdjęcia RGB. Wydzielenie z wczytanego obrazu warstwy w kolorze czerwonym do detekcji źrenicy.
 *  3. Detekcja źrenicy: progowanie typu truncate (obcięcie wysokich wartości pikseli).
 *  4. Detekcja źrenicy: wykonanie operacji dilate i erode.
 *  5. Detekcja źrenicy: Rozmycie Gaussa.
 *  6. Detekcja źrenicy: Transformata Hougha (ograniczenie minimalnego promienia okręgu uniewrażliwia na detekcję refleksów światła.
 *  7. Naniesienie źrenicy na obraz.
 *  8. Zmiana obrazu RGB na GRAY do detekcja limbusa.
 *  9. Detekcja krawędzi limbusa - piksel prawej krawędzi ma po swojej prawej piksele o znacznie większej średniej wartości, 
 *          a po lewej piksele o podobnej średniej wartości. Analogiczna procedura dla pikseli lewej krawędzi. Otrzymanie obrazu 
 *          krawędzi limbusa.
 *  10. Selekcja krawędzi limbusa na obrazie - przypisanie pikselom położonym pomiędzy kątami 30 - 150st oraz 225 - 315st względem
 *          środka źrenicy wartości 0. Na obrazie pozostają tylko piksele, które leżą w zakresie kątowym odpowiadającym "pionowym" 
 *          granicom limbusa.
 *  11. Detekcja okręgów na obrazie krawędzi limbusa poprzez zastosowanie własnej funkcji, wzorowanej na transformacie Hougha.
 *  12. Naniesienie limbusa na obraz.

 */
#include <iostream>
#include <cstdlib>
#include <opencv2/opencv.hpp>
 
using namespace std;
using namespace cv;


int main()
{
    // reading all images from folder, in for loop
    vector<cv::String> fn;
    glob("/home/stanley/Pulpit/praca_mgr/praca_mgr/program8/images/*.jpg", fn, false);
    vector<Mat> images;
    size_t count = fn.size(); //number of png files in images folder
    for (size_t i=0; i<count; i++)
    {
        Mat src, pupilRed, resultImage;
        //src=imread("oko10.jpg");
        src=imread(fn[i]);
        resultImage = src;
        Mat bgr[3];   //destination array
        split(src,bgr);//split source
        /*
        namedWindow("src image", WINDOW_AUTOSIZE);
        imshow("src image", src);
        namedWindow("Blue", WINDOW_AUTOSIZE);
        imshow("Blue", bgr[0]);
        namedWindow("Green", WINDOW_AUTOSIZE);
        imshow("Green", bgr[1]);
        namedWindow("Red", WINDOW_AUTOSIZE);
        imshow("Red", bgr[2]);
        */
        
        //////////////// PUPIL DETECTING
        pupilRed=bgr[2];    //red layer, pupil detecting

        threshold(pupilRed, pupilRed, 80, 255, 2);
        // element for specified dilation and erosion
        Mat element = getStructuringElement( MORPH_RECT,
                                        Size( 7, 7 ));
        Mat element2 = getStructuringElement( MORPH_RECT,
                                        Size( 5, 5 ));
        
        dilate(pupilRed, pupilRed, element);
        erode(pupilRed, pupilRed, element2);
    
        GaussianBlur(pupilRed, pupilRed, Size(11,11), 2, 2 );
        
        namedWindow("pupilRed", WINDOW_AUTOSIZE);
        imshow("pupilRed", pupilRed);
   
        // circles detection - pupil
        vector<Vec3f> circlesPupil;
        HoughCircles( pupilRed, circlesPupil, CV_HOUGH_GRADIENT, 2, pupilRed.rows, 25, 15, 
                  pupilRed.rows/20, pupilRed.rows/6);

        // drawing a pupil circle on result image
        Point center(cvRound(circlesPupil[0][0]), cvRound(circlesPupil[0][1]));
        int radiusPupil = cvRound(circlesPupil[0][2]);
        circle( resultImage, center, 3, Scalar(0,0,255), -1, 8, 0 );
        circle( resultImage, center, radiusPupil, Scalar(0,0,255), 1, 8, 0 );
        
        //////////////// LIMBUS DETECTING
        
        cvtColor(src, src, CV_BGR2GRAY);
        // center of pupil
        int xp = circlesPupil[0][0]; //column
        int yp = circlesPupil[0][1]; //row
    
        // image of limbus edges
        Mat limbusEdgeImage (src.rows, src.cols, src.type());
        limbusEdgeImage = 0;
        int neighbours = 50; //number od pixels on the left and right 
        
        //this loop is searching edges on the left of pupil
        for (int i=neighbours; i<xp; i++)
        {
            for (int j=0; j<src.rows; j++)
            {
                float meanRight = 0;
                float meanLeft = 0;
                for (int k=0; k<neighbours; k++)
                {
                    //mean values od pixels on the right and left
                    meanRight += src.at<uchar>(j, i+k);
                    meanLeft += src.at<uchar>(j, i-k);
                }
                meanRight = meanRight / neighbours;
                meanLeft = meanLeft / neighbours;
                int actualValue = src.at<uchar>(j, i);
                if ( (abs(meanRight - actualValue) < 15 ) && (abs(meanLeft - actualValue) > 50) ) 
                    limbusEdgeImage.at<uchar>(j, i) = 255;
                else 
                    limbusEdgeImage.at<uchar>(j, i) = 0;
            }
        }
    
        //this loop is searching edges on the right of pupil
        for (int i=xp; i<src.cols-neighbours; i++)
        {
            for (int j=0; j<src.rows; j++)
            {
                float meanRight = 0;
                float meanLeft = 0;
                for (int k=0; k<neighbours; k++)
                {
                    meanRight += src.at<uchar>(j, i+k);
                    meanLeft += src.at<uchar>(j, i-k);
                }
                meanRight = meanRight / neighbours;
                meanLeft = meanLeft / neighbours;
                int actualValue = src.at<uchar>(j, i);
                if ( (abs(meanLeft - actualValue) < 15 ) && (abs(meanRight - actualValue) > 50) )
                    limbusEdgeImage.at<uchar>(j, i) = 255;
                else
                    limbusEdgeImage.at<uchar>(j, i) = 0;
            }
        }
    
        //namedWindow("limbusEdgeImage", WINDOW_AUTOSIZE);
        //imshow("limbusEdgeImage", limbusEdgeImage);

        
        // deleting pixels in 30-150st and 225-315st
    
        // deleting on the left of pupil
        for (int i=0; i<xp; i++)
        {
            for (int j=0; j<limbusEdgeImage.rows; j++)
            {
                float tangens = (float)(-j + yp) / (xp - i);
                if ((tangens<-1) || (tangens>0.577))
                    limbusEdgeImage.at<uchar>(j, i) = 0;
            }
        }
        
        // deleting on the right of pupil
        for (int i=xp; i<limbusEdgeImage.cols; i++)
        {
            for (int j=0; j<limbusEdgeImage.rows; j++)
            {
                float tangens = (float)(-j + yp) / (xp - i);
                if ((tangens<-0.577) || (tangens>1))
                    limbusEdgeImage.at<uchar>(j, i) = 0;
            }
        }
    
        //namedWindow("limbusEdgeImage", WINDOW_AUTOSIZE);
        //imshow("limbusEdgeImage", limbusEdgeImage);
    
    
        // circles detection - own function, without canny
    
        erode(limbusEdgeImage, limbusEdgeImage, Mat());  //deleting single pixels
        namedWindow("limbusEdgeImage", WINDOW_AUTOSIZE);
        imshow("limbusEdgeImage", limbusEdgeImage);

        // vectors - table with informaction about detected circles
        vector <int> centerX, centerY, rLimb, power;
        
        // a loop which detecting circles with center placed in pupil
        for (int r=(int)1.25 * radiusPupil; r<limbusEdgeImage.rows / 2; r+=4)
        {
            // image with circles from many pixels
            Mat circles (limbusEdgeImage.rows, limbusEdgeImage.cols, limbusEdgeImage.type());
            circles = 0;
            
            // vectors with informaction about position of non-zero pixels
            vector <int> xFill;
            vector <int> yFill; 

            int iter = 0;
            for (int x=0; x<limbusEdgeImage.cols; x++)
            {
                for (int y=0; y<limbusEdgeImage.rows; y++)
                {
                    if (limbusEdgeImage.at<uchar>(y , x) != 0)
                    {
                        xFill.push_back(x);
                        yFill.push_back(y);
                        iter++;
                    }
                }
            }
       
            // random numbers
            int numRand = 200;
            int random[numRand] = {0};
            for (int i=0; i<numRand; i++)
                random[i] = rand() % iter;
        
            //drawing circles from random pixels
            for (int i=0; i<numRand; i++)
            {
                Point center(xFill[random[i]], yFill[random[i]]);
                int radius = r;
                Mat temporary (limbusEdgeImage.rows, limbusEdgeImage.cols, limbusEdgeImage.type());
                temporary = 0;
                circle( temporary, center, radius, Scalar(1,0,0), 1, 8, 0 );
                circles += temporary;
            }

            int actualValue = 0, centerx, centery;  //parameters of the best circle for actual radius r (from loop) 
            for (int x=0; x<limbusEdgeImage.cols; x++)
            {
                for (int y=0; y<limbusEdgeImage.rows; y++)
                {
                    if (circles.at<uchar>(y , x) > actualValue)
                    {
                        centerx = x;
                        centery = y;
                        actualValue = circles.at<uchar>(y , x);
                    }
                }
            }
                
            //remember this circle, if the center is in the pupil
            if ( pow(xp - centerx, 2) + pow(yp - centery, 2) < pow(radiusPupil, 2))
            {
                centerX.push_back(centerx);
                centerY.push_back(centery);
                rLimb.push_back(r);
                power.push_back(actualValue);
            }
        }
    
        // choosing the best circle
        int maxPower = 0;
        int z = 0;
        for( size_t i = 0; i < power.size(); i++ )
        {
            if (power[i] > maxPower)
            {
                maxPower = power[i];
                z=i;
            }
        }
    
        // drawing limbus circle 
        Point centerLimb(centerX[z], centerY[z]);
        int radiusLimb = rLimb[z];
        circle( resultImage, centerLimb, radiusLimb, Scalar(255,0,0), 1, 8, 0 );
    
        // view the result image
        namedWindow("resultImage", WINDOW_AUTOSIZE);
        imshow("resultImage", resultImage);

        waitKey(0);
    
        // saving result images
        String im_name = fn[i]+"_edit";
        cout<<im_name<<endl;
        imwrite(im_name, resultImage);
    }
    return 0;
}
