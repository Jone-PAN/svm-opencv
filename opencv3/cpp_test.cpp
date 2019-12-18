#include <iostream>
#include <opencv2/opencv.hpp>

#define WIDTH 16
#define HEIGHT 16

int main(int argc, char **argv) {

  if (argc != 2) {
    std::cout << "Usage: ./svm_test <input_image>\n";
    return -1;
  }
  char *image_path = argv[1];

  cv::Ptr<cv::ml::SVM> svm = cv::Algorithm::load<cv::ml::SVM>("svm.xml");
  if (!svm) {
    std::cout << "Unable to load SVM model!" << std::endl;
    return -1;
  }
  std::cout << "SVM model is loaded.\n";

  cv::HOGDescriptor hog;
  hog.load("hog_descriptor.xml");
  std::cout << "HOG Descriptor model is loaded.\n";

  std::cout << "hog.blockSize: " << hog.blockSize << "\n"
            << "hog.blockStride: " << hog.blockStride << "\n"
            << "hog.winSigma: " << hog.winSigma << "\n"
            << "hog.winSize: " << hog.winSize << "\n";

  cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
  if (image.empty()) {
    std::cout << "Unable to read the input image: " << image_path << std::endl;
    return -1;
  }
  cv::resize(image, image, cv::Size(WIDTH, HEIGHT));
  std::cout << "Image is loaded.\n";

  std::vector<float> hog_descriptors;
  hog.compute(image, hog_descriptors);
  int size = hog_descriptors.size();
  cv::Mat test_mat(1, size, CV_32FC1);
  for (size_t i = 0; i < hog_descriptors.size(); ++i) {
    test_mat.at<float>(0, i) = hog_descriptors[i];
  }

  std::cout << "Start to predict using SVM model ...\n";
  cv::Mat test_result;
  svm->predict(test_mat, test_result);
  for (int i = 0; i < test_result.rows; ++i) {
    for (int j = 0; j < test_result.cols; ++j) {
      std::cout << "SVM predict result for this image is: "
                << test_result.at<float>(i, j) << std::endl;
    }
  }

  return 0;
}