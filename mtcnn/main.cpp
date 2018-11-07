#include "DetectorFace.hpp"
#include "Timer.hpp"


int main(int argc, char *argv[]) {
	string model_path;
	string image_path;
	if (argc == 3) {
		model_path = argv[1];
		image_path = argv[2];
	} else {
		model_path = "../models/mtcnn/mtcnn.pb";
		image_path = "../data/face.jpg";
	}
	facedetector detector(model_path);
	Timer timer;
	timer.Tic();
	detector.detect(image_path);
	timer.Toc();
	std::cout << "cost time:" << timer.Elasped() << "ms" << std::endl;

	detector.draw_bbox_image();

	return 0;
}