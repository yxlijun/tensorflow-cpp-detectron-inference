#include "Detectorsdd.hpp"

int main(int argc, char *argv[]) {
	string model_path;
	string image_path;
	if (argc == 3) {
		model_path = argv[1];
		image_path = argv[2];
	} else {
		model_path = "../models/ssd/ssd300.pb";;
		image_path =  "../data/dog.jpg";
	}
	Detectorsdd detector(model_path);
	detector.detect(image_path);
	detector.draw_bbox_image();
	return 0;
}