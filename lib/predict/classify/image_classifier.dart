import 'dart:typed_data';

import 'package:ultralytics_yolo/predict/classify/classification_result.dart';
import 'package:ultralytics_yolo/predict/predictor.dart';
import 'package:ultralytics_yolo/yolo_model.dart';

/// A predictor for image classification.
class ImageClassifier extends Predictor {
  /// Constructor to create an instance of [ImageClassifier].
  ImageClassifier({required YoloModel model}) : super(model);

  /// The platform instance used to run image classification.
  Stream<List<ClassificationResult?>?> get classificationResultStream =>
      ultralyticsYoloPlatform.classificationResultStream;

  /// Classifies an image from the given [imagePath] or [imageBytes].
  Future<List<ClassificationResult?>?> classify({
    String? imagePath,
    Uint8List? imageBytes,
  }) =>
      ultralyticsYoloPlatform.classifyImage(model.id, imagePath, imageBytes);
}
