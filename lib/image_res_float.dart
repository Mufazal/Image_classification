import 'package:imageclassification/image_res.dart';
import 'package:tflite_flutter_helper/tflite_flutter_helper.dart';

class ImageResolutionFloat extends ImageResolutionTensor {
  ImageResolutionFloat({int numThreads = 1}) : super(numThreads: numThreads);

  @override
  String get modelName => 'lite_model_esrgan.tflite';

  @override
  NormalizeOp get preProcessNormalizeOp => NormalizeOp(127.5, 127.5);

  @override
  NormalizeOp get postProcessNormalizeOp => NormalizeOp(0, 1);

  // ///
  // ///
  // //Quant
  // @override
  // NormalizeOp get preProcessNormalizeOp => NormalizeOp(0, 1);

  // @override
  // NormalizeOp get postProcessNormalizeOp => NormalizeOp(0, 255);
}
