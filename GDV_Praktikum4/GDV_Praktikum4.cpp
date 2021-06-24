#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

cv::Mat plotHistogram(cv::Mat& image, bool cumulative = false, int histSize = 256);

static std::string BASE_PATH = "C:\\Users\\lars\\Documents\\Uni\\SS21\\gdv\\Praktikum4";


void aufgabe1() {
    cv::Mat img = cv::imread(BASE_PATH + "\\schrott.png"); // Read the file
    if (img.empty()) // Check for invalid input
    {
        std::cout << "Could not open or find the frame" << std::endl;
        return;
    }

    cv::Mat img_gray, img_gray2;
    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);    // In case img is colored
    cv::cvtColor(img, img_gray2, cv::COLOR_BGR2GRAY);

    double min, max;
    cv::minMaxLoc(img_gray, &min, &max);
    std::cout << "Min: " << min << " Max: " << max << '\n';

    // values for brightness transformation 
    double d = 100;
    double k = (150 - 100) / max;
    img_gray = (k * img_gray) + d;

    d = 0;
    k = (255) / max;
    img_gray2 = (k * img_gray2) + d;

    cv::namedWindow("Input Image", cv::WINDOW_AUTOSIZE); // Create a window for display.
    cv::imshow("Input Image", img_gray);
    cv::namedWindow("Input Image2", cv::WINDOW_AUTOSIZE); // Create a window for display.
    cv::imshow("Input Image2", img_gray2);
    cv::Mat hist, hist2;
    hist = plotHistogram(img_gray, true);
    cv::namedWindow("Histogram", cv::WINDOW_AUTOSIZE); // Create a window for display.
    cv::imshow("Histogram", hist);
    hist2 = plotHistogram(img_gray2, true);
    cv::namedWindow("Histogram2", cv::WINDOW_AUTOSIZE); // Create a window for display.
    cv::imshow("Histogram2", hist2);

    cv::imwrite(BASE_PATH + "\\no_dynamic.png", img_gray2);
}

int aufgabe2() {
    cv::Mat img = cv::imread(BASE_PATH + "\\schrott.png"); // Read the file
    if (img.empty()) // Check for invalid input
    {
        std::cout << "Could not open or find the frame" << std::endl;
        return -1;
    }

    cv::Mat img_gray, img_gray2;
    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);    // In case img is colored
    cv::cvtColor(img, img_gray2, cv::COLOR_BGR2GRAY);    // In case img is colored

    double min, max;
    cv::minMaxLoc(img_gray, &min, &max);
    //std::cout << "Min: " << min << " Max: " << max << '\n';

    cv::equalizeHist(img_gray, img_gray);

    cv::namedWindow("Aufgabe2", cv::WINDOW_AUTOSIZE); // Create a window for display.
    cv::imshow("Aufgabe2", img_gray);
    
    int totalPixels = img_gray.cols * img_gray.rows;

    // Schritt 1
    int hist[256];
    memset(hist, 0, sizeof(hist));
    for (int i = 0; i < img_gray.rows; i++) {
        for (int j = 0; j < img_gray.cols; j++) {
            int level = img_gray.at<uchar>(i, j);
            hist[level]++;
        }
    }

    // Schritt 2 
    int hist_acc[256];
    memset(hist_acc, 0, sizeof(hist_acc));
    hist_acc[0] = hist[0];
    for (int i = 1; i < 256; i++) {
        hist_acc[i] = hist[i] + hist_acc[i - 1];
    }

    // Schritt 3 
    double cdf[256];
    memset(cdf, 0, sizeof(cdf));
    for (int i = 0; i < 256; i++) {
        cdf[i] = (double)hist_acc[i] / (double)totalPixels;
    }

    // Schritt 4 
    int rd[256];
    memset(rd, 0, sizeof(rd));
    for (int i = 0; i < 256; i++) {
        rd[i] = cvRound(cdf[i] * 255);
    }

    // Schritt 5
    for (int i = 0; i < img_gray.rows; i++) {
        for (int j = 0; j < img_gray.cols; j++) {
            int pixelValue = img_gray.at<uchar>(i, j);
            img_gray2.at<uchar>(i, j) = rd[pixelValue];
        }
    }
    cv::namedWindow("Aufgabe2 own", cv::WINDOW_AUTOSIZE); // Create a window for display.
    cv::imshow("Aufgabe2 own", img_gray2);
    cv::Mat hist2 = plotHistogram(img_gray2, true);
    cv::namedWindow("Histogram Aufgabe2", cv::WINDOW_AUTOSIZE); // Create a window for display.
    cv::imshow("Histogram Aufgabe2", hist2);
}

void aufgabe3() {
    cv::Mat kante = cv::imread(BASE_PATH + "\\kante.png"); // Read the file
    if (kante.empty()) // Check for invalid input
    {
        std::cout << "Could not open or find the frame" << std::endl;
        return;
    }
    // -------------------------------------------------------------------
    // Definition Matrizen 

    /*+++++++++++++++++++++++++++++++++++++++*/
    /*                   F1                  */
    /*+++++++++++++++++++++++++++++++++++++++*/
    cv::Mat f1 = cv::Mat::zeros(3, 3, CV_32F);
    f1.at<float>(1, 0) = -1;
    f1.at<float>(1, 1) = 1;
    std::cout << "F1:\n";
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            std::cout << (int)f1.at<float>(i, j) << '\t';
        }
        std::cout << '\n';
    }

    /*+++++++++++++++++++++++++++++++++++++++*/
    /*                   F2                  */
    /*+++++++++++++++++++++++++++++++++++++++*/
    cv::Mat f2 = cv::Mat::zeros(3, 3, CV_32F);
    f2.at<float>(1, 1) = -1;
    f2.at<float>(1, 2) = 1;
    std::cout << "F2:\n";
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            std::cout << (int)f2.at<float>(i, j) << '\t';
        }
        std::cout << '\n';
    }

    /*+++++++++++++++++++++++++++++++++++++++*/
    /*                   f3                  */
    /*+++++++++++++++++++++++++++++++++++++++*/
    cv::Mat f3 = cv::Mat::zeros(3, 3, CV_32F);
    f3.at<float>(1, 0) = 1;
    f3.at<float>(1, 1) = -2;
    f3.at<float>(1, 2) = 1;
    std::cout << "f3:\n";
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            std::cout << (int)f3.at<float>(i, j) << '\t';
        }
        std::cout << '\n';
    }

    /*+++++++++++++++++++++++++++++++++++++++*/
    /*                   f4                  */
    /*+++++++++++++++++++++++++++++++++++++++*/
    cv::Mat f4 = cv::Mat::zeros(3, 3, CV_32F);
    f4.at<float>(1, 0) = (1.0 / 3);
    f4.at<float>(1, 1) = (1.0 / 3);
    f4.at<float>(1, 2) = (1.0 / 3);
    std::cout << "f4:\n";
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            std::cout << f4.at<float>(i, j) << "\t\t";
        }
        std::cout << '\n';
    }

    /*+++++++++++++++++++++++++++++++++++++++*/
    /*                   f5                  */
    /*+++++++++++++++++++++++++++++++++++++++*/
    cv::Mat f5 = cv::Mat::zeros(3, 3, CV_32F);
    f5.at<float>(1, 1) = 1;
    
    std::cout << "f5:\n";
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            std::cout << f5.at<float>(i, j) << "\t\t";
        }
        std::cout << '\n';
    }

    /*+++++++++++++++++++++++++++++++++++++++*/
    /*                   f6                  */
    /*+++++++++++++++++++++++++++++++++++++++*/
    cv::Mat f6 = cv::Mat::zeros(3, 3, CV_32F);
    f6.at<float>(1, 0) = (1.0 / 3);
    f6.at<float>(1, 1) = (-2.0 / 3);
    f6.at<float>(1, 2) = (1.0 / 3);
    std::cout << "f6:\n";
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            std::cout << f6.at<float>(i, j) << "\t\t";
        }
        std::cout << '\n';
    }
    cv::namedWindow("Aufgabe3 Kante.png", cv::WINDOW_GUI_NORMAL); // Create a window for display.
    cv::resizeWindow("Aufgabe3 Kante.png", 280, 160);
    cv::imshow("Aufgabe3 Kante.png", kante);
    // -------------------------------------------------------------------
    // Faltung mit F1 
    cv::Mat dst;
    cv::filter2D(kante, dst, -1, f1, cv::Point(-1,-1), 128);

    cv::namedWindow("Aufgabe3 F1", cv::WINDOW_GUI_NORMAL ); // Create a window for display.
    cv::resizeWindow("Aufgabe3 F1", 280, 160);
    cv::imshow("Aufgabe3 F1", dst);

    // -------------------------------------------------------------------
    // Faltung mit F2 auf Ergebnis von F1
    cv::filter2D(dst, dst, -1, f2, cv::Point(-1, -1), 128);
    cv::namedWindow("Aufgabe3 F2", cv::WINDOW_GUI_NORMAL); // Create a window for display.
    cv::resizeWindow("Aufgabe3 F2", 280, 160);
    cv::imshow("Aufgabe3 F2", dst);
    
    // -------------------------------------------------------------------
    // Faltung mit F3
    cv::Mat dst2;
    cv::filter2D(kante, dst2, -1, f3, cv::Point(-1, -1), 128);

    cv::namedWindow("Aufgabe3 F3", cv::WINDOW_GUI_NORMAL); // Create a window for display.
    cv::resizeWindow("Aufgabe3 F3", 280, 160);
    cv::imshow("Aufgabe3 F3", dst2);

    // -------------------------------------------------------------------
    // Faltung mit F4
    cv::Mat dst3;
    cv::filter2D(kante, dst3, -1, f4, cv::Point(-1, -1));

    cv::namedWindow("Aufgabe3 F4", cv::WINDOW_GUI_NORMAL); // Create a window for display.
    cv::resizeWindow("Aufgabe3 F4", 280, 160);
    cv::imshow("Aufgabe3 F4", dst3);

    // -------------------------------------------------------------------
    // Faltung mit F5
    cv::Mat dst4;
    cv::filter2D(kante, dst4, -1, f5, cv::Point(-1, -1));

    cv::namedWindow("Aufgabe3 F5", cv::WINDOW_GUI_NORMAL); // Create a window for display.
    cv::resizeWindow("Aufgabe3 F5", 280, 160);
    cv::imshow("Aufgabe3 F5", dst4);
    
    // -------------------------------------------------------------------
    // Faltung mit F6
    cv::Mat dst5;
    cv::filter2D(kante, dst5, -1, f6, cv::Point(-1, -1), 128);

    cv::namedWindow("Aufgabe3 F6", cv::WINDOW_GUI_NORMAL); // Create a window for display.
    cv::resizeWindow("Aufgabe3 F6", 280, 160);
    cv::imshow("Aufgabe3 F6", dst5);
    waitKey(0);

    /*
        F1) Das Ergebnis der beiden letzten Faltungen ist identisch. 
        F2) F3 bildet die zweite Ableitung in eine Richtung (horizontal)
        F3) Das Ergebnis ist prinzipiell das gleiche, nur ist der Kontrast von F6 nur ein drittel des Kontrastes von F3
        F4) F4 nimmt den Durschnitt der drei horizontal liegenden Pixel. Somit eine Glättung in x-Richtung. 
            F5 ist der Identitätsoperator und mach "nichts".
        F5) Der LaPlace Operator, ist die Summe der zweiten Ableitungen. Er ist ein Kantenfilter.
            [0, 1, 0]
            [1,-4, 1]
            [0, 1, 0]
    */

}

int main()
{
    aufgabe1();
    aufgabe2();
    aufgabe3();
    return 0;
}



cv::Mat plotHistogram(cv::Mat& image, bool cumulative, int histSize) {
    // Create Image for Histogram
    int hist_w = 1024; int hist_h = 800;
    int bin_w = cvRound((double)hist_w / histSize);

    cv::Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(255, 255, 255));

    if (image.channels() > 1) {
        cerr << "plotHistogram: Please insert only gray images." << endl;
        return histImage;
    }

    // Calculate Histogram
    float range[] = { 0, 256 };
    const float* histRange = { range };

    cv::Mat hist;
    calcHist(&image, 1, 0, Mat(), hist, 1, &histSize, &histRange);

    if (cumulative) {
        cv::Mat accumulatedHist = hist.clone();
        for (int i = 1; i < histSize; i++) {
            accumulatedHist.at<float>(i) += accumulatedHist.at<float>(i - 1);
        }
        hist = accumulatedHist;
    }

    // Normalize the result to [ 0, histImage.rows ]
    normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

    // Draw bars
    for (int i = 1; i < histSize; i++) {
        cv::rectangle(histImage, Point(bin_w * (i - 1), hist_h),
            Point(bin_w * (i), hist_h - cvRound(hist.at<float>(i))),
            Scalar(50, 50, 50), 1);
    }

    return histImage;   // Not really call by value, as cv::Mat only saves a pointer to the image data
}
