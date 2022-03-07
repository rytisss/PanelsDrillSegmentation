#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/core/operations.hpp> //partitions
#include <string>
#include <vector>
#include <filesystem>
#include <iostream>

void make_directory(std::string directory)
{
	std::filesystem::path p(directory);
	std::filesystem::create_directory(p);
}

std::vector<std::string> GetImageFromDirectory(std::string directory)
{
	std::vector<std::string> images;
	std::string path = directory + "/*.png";
	cv::glob(directory, images, false);
	return images;
}


cv::Mat ClusterContours(cv::Mat& image, bool findContours = true)
{
	// DO CLUSTERING BY DISTANCE
	int th_distance = 5; // radius tolerance
	int th2 = th_distance * th_distance; // squared radius tolerance
	std::vector<int> labels;
	//find contours in image and make labels
	std::vector<std::vector<cv::Point>> contours;
	if (findContours)
	{
		cv::findContours(image, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
	}
	else
	{

	}

	cv::Mat clusteredImage = cv::Mat(image.rows, image.cols, CV_8UC1);
	clusteredImage.setTo(0);//all black
	if (contours.size() > 0)
	{
		//make single contour list
		std::vector<cv::Point> contours_points;
		for (size_t contour_index = 0; contour_index < contours.size(); contour_index++)
		{
			for (size_t pt_index = 0; pt_index < contours[contour_index].size(); pt_index++)
			{
				contours_points.push_back(contours[contour_index][pt_index]);
			}
		}
		int n_labels = cv::partition(contours_points, labels, [th2](const cv::Point& lhs, const cv::Point& rhs) {
			return ((lhs.x - rhs.x) * (lhs.x - rhs.x) + (lhs.y - rhs.y) * (lhs.y - rhs.y)) < th2;
			});

		//form new clustered contours
		std::vector<std::vector<cv::Point>> clusteredContours;
		//make as much contour as we have labels
		for (size_t i = 0; i < n_labels; i++)
		{
			clusteredContours.push_back(std::vector<cv::Point>());
		}
		//put points to clustered contour vector
		for (size_t i = 0; i < contours_points.size(); i++)
		{
			int clusterIndex = labels[i];
			clusteredContours[clusterIndex].push_back(contours_points[i]);
		}

		//form new image with clustered closed contours and filter out too big or too small contour

		//5 pixels
		double min_area = 5.0;
		//25% of image
		double max_area = (double)(image.cols * image.rows) * 0.2;
	
		
		std::vector<std::vector<cv::Point>> hulls(clusteredContours.size());
		for (size_t i = 0; i < clusteredContours.size(); i++)
		{
			double contourSize = cv::contourArea(clusteredContours[i]);
			if (contourSize > min_area && contourSize < max_area)
			{
				//close contour with convex hull
				cv::convexHull(clusteredContours[i], hulls[i]);
				cv::drawContours(clusteredImage, hulls, i, cv::Scalar(255), cv::FILLED);
			}
		}
	}
	return clusteredImage;
}


int GetMaxValue(cv::Mat& image)
{
	double minVal;
	double maxVal;
	cv::Point minLoc;
	cv::Point maxLoc;
	cv::minMaxLoc(image, &minVal, &maxVal, &minLoc, &maxLoc);
	return (int)(maxVal);
}

cv::Mat GetThEdges(cv::Mat& image, float th = 0.75f)
{
	//sanity check
	if (th < 0.f)
	{
		th = 0.f;
	}
	if (th > 1.f)
	{
		th = 1.f;
	}
	int maxLimit = GetMaxValue(image);
	int limit = (int)((float)maxLimit * th);
	cv::Mat threshold;
	cv::threshold(image, threshold, limit, 255, cv::THRESH_BINARY);
	return threshold;
}

cv::Mat GetCanny(cv::Mat& image)
{
	cv::Mat edges;
	cv::Canny(image, edges, 150, 200);
	return edges;
}


cv::Mat GetLaplace(cv::Mat& image)
{
	int kernel_size = 3;
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;
	cv::Mat dst;
	cv::Laplacian(image, dst, ddepth, kernel_size, scale, delta, cv::BORDER_DEFAULT);
	// converting back to CV_8U
	cv::Mat laplacian;
	convertScaleAbs(dst, laplacian);
	return laplacian;
}


cv::Mat GetSobel(cv::Mat& image)
{
	cv::Mat grad_x, grad_y;
	cv::Mat abs_grad_x, abs_grad_y;
	int ksize = 3;
	int ddepth = CV_16S;
	cv::Sobel(image, grad_x, ddepth, 1, 0, ksize, 1, 0, cv::BORDER_DEFAULT);
	cv::Sobel(image, grad_y, ddepth, 0, 1, ksize, 1, 0, cv::BORDER_DEFAULT);
	// converting back to CV_8U
	convertScaleAbs(grad_x, abs_grad_x);
	convertScaleAbs(grad_y, abs_grad_y);
	cv::Mat grad;
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
	return grad;
}


void ProcessImage(std::string path)
{
	//get filename
	std::filesystem::path p(path);
	std::string imageName = p.filename().string();
	std::cout << imageName << std::endl;
	//open image
	cv::Mat image = cv::imread(path, cv::ImreadModes::IMREAD_GRAYSCALE);
	//cv::imshow("image", image);
	//Sobel
	cv::Mat sobel = GetSobel(image);
	cv::Mat sobelTh = GetThEdges(sobel);
	//cv::imshow("Sobel_th", sobelTh);
	make_directory("Sobel_th/");
	cv::imwrite("Sobel_th/" + imageName + ".png", sobelTh);
	cv::Mat sobelClustered = ClusterContours(sobelTh);
	//cv::imshow("sobel clustered", sobelClustered);
	make_directory("Sobel_clustered/");
	cv::imwrite("Sobel_clustered/" + imageName + ".png", sobelClustered);
	//Laplace
	cv::Mat laplace = GetLaplace(image);
	cv::Mat laplaceTh = GetThEdges(laplace);
	//cv::imshow("laplace_th", laplaceTh);
	make_directory("Laplace_th/");
	cv::imwrite("Laplace_th/" + imageName + ".png", laplaceTh);
	cv::Mat laplaceClustered = ClusterContours(laplaceTh);
	//cv::imshow("laplace clustered", laplaceClustered);
	make_directory("Laplace_clustered/");
	cv::imwrite("Laplace_clustered/" + imageName + ".png", laplaceClustered);
	//Canny
	cv::Mat canny = GetCanny(image);
	//cv::imshow("canny", canny);
	make_directory("Canny_th/");
	cv::imwrite("Canny_th/" + imageName + ".png", laplaceTh);
	cv::Mat cannyClustered = ClusterContours(canny);
	make_directory("Canny_clustered/");
	cv::imwrite("Canny_clustered/" + imageName + ".png", cannyClustered);
	//cv::imshow("canny clustered", cannyClustered);

	//cv::waitKey(1);
}


int main()
{
	std::string path = "data_dir/";
	std::vector<std::string> imagePaths = GetImageFromDirectory(path);
	for (size_t i = 0; i < imagePaths.size(); i++)
	{
		ProcessImage(imagePaths[i]);
	}
	return 0;
}
