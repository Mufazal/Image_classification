import 'dart:math';
import 'dart:typed_data';

import 'package:collection/collection.dart';
import 'package:image/image.dart' as img;
import 'package:logger/logger.dart';
//import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:tflite_flutter/tflite_flutter.dart' as tf;
import 'package:tflite_flutter_helper/tflite_flutter_helper.dart';

abstract class ImageResolutionTensor {
  tf.Interpreter interpreter;
  tf.InterpreterOptions _interpreterOptions;

  var logger = Logger();

  List<int> _inputShape;
  List<int> _outputShape;

  TensorBuffer _outputBuffer;

  tf.TfLiteType _outputType = tf.TfLiteType.uint8;
  TensorImage _inputImage; //= TensorImage(_outputType);
  final String _labelsFileName = 'assets/labels.txt';

  final int _labelsLength = 1001;

  var _probabilityProcessor;

  List<String> labels;

  String get modelName;

  NormalizeOp get preProcessNormalizeOp;

  NormalizeOp get postProcessNormalizeOp;

  ImageResolutionTensor({int numThreads}) {
    final gpuDelegateV2 = tf.GpuDelegateV2(
        options: tf.GpuDelegateOptionsV2(
      false,
      tf.TfLiteGpuInferenceUsage.fastSingleAnswer,
      tf.TfLiteGpuInferencePriority.minLatency,
      tf.TfLiteGpuInferencePriority.auto,
      tf.TfLiteGpuInferencePriority.auto,
    ));

    _interpreterOptions = tf.InterpreterOptions()..useNnApiForAndroid = true;
    // ..addDelegate(gpuDelegateV2);

    if (numThreads != null) {
      _interpreterOptions.threads = numThreads;
    }

    loadModel();
    loadLabels();
  }

  Future<void> loadModel() async {
    try {
      interpreter = await tf.Interpreter.fromAsset(modelName,
          options: _interpreterOptions);
      print('Interpreter Created Successfully');

      _inputShape = interpreter.getInputTensor(0).shape;
      _outputShape = interpreter.getOutputTensor(0).shape;
      _outputType = interpreter.getOutputTensor(0).type;
      interpreter.allocateTensors();
      _outputBuffer = TensorBuffer.createFixedSize(_outputShape, _outputType);
      _probabilityProcessor =
          TensorProcessorBuilder().add(postProcessNormalizeOp).build();
    } catch (e) {
      print('Unable to create interpreter, Caught Exception: ${e.toString()}');
    }
  }

  Future<void> loadLabels() async {
    labels = await FileUtil.loadLabels(_labelsFileName);
    if (labels.length == _labelsLength) {
      print('Labels loaded successfully');
    } else {
      print('Unable to load labels');
    }
  }

  TensorImage _preProcess() {
    print("Inpit shape ------------1--$_inputShape[1]\n");
    print("Inpit shape ------------2--$_inputShape[2]\n");
    int cropSize = min(_inputImage.height, _inputImage.width);
    return ImageProcessorBuilder()
        .add(ResizeWithCropOrPadOp(cropSize, cropSize))
        .add(ResizeOp(_inputShape[1], _inputShape[2], ResizeMethod.BILINEAR))
        .add(preProcessNormalizeOp)
        .build()
        .process(_inputImage);
  }

  Uint8List predict(img.Image image, {int height, int width}) {
    if (interpreter == null) {
      throw StateError('Cannot run inference, Intrepreter is null');
    }
    final pres = DateTime.now().millisecondsSinceEpoch;
    _inputImage = TensorImage.fromImage(image);
    _inputImage = _preProcess();
    final pre = DateTime.now().millisecondsSinceEpoch - pres;

    print('Time to load image: $pre ms');
    // print("OUT Put Buffer 1 : ${_outputBuffer.buffer.asUint8List()}");
    var outputImageData = [
      List.generate(
        200,
        (index) => List.generate(
          200,
          (index) => List.generate(3, (index) => 0.0),
        ),
      ),
    ];
    final runs = DateTime.now().millisecondsSinceEpoch;
    interpreter.run(_inputImage.buffer, outputImageData);
    final run = DateTime.now().millisecondsSinceEpoch - runs;

    print(_outputBuffer.buffer.lengthInBytes);

    //  print(_outputBuffer.buffer.asUint8List());
    print('Time to run inference: $run ms');

    // Map<String, double> labeledProb = TensorLabel.fromList(
    //         labels, _probabilityProcessor.process(_outputBuffer))
    //     .getMapWithFloatValue();
    // final pred = getTopProbability(labeledProb);

    var outputImage = _convertArrayToImage(outputImageData, 200);
    var rotateOutputImage = img.copyRotate(outputImage, 90);
    var flipOutputImage = img.flipHorizontal(rotateOutputImage);
    var resultImage = img.copyResize(flipOutputImage, width: 200, height: 200);
    return img.encodeJpg(resultImage);
    // return Category(pred.key, pred.value);
  }

  void close() {
    if (interpreter != null) {
      interpreter.close();
    }
  }
}

img.Image _convertArrayToImage(
    List<List<List<List<double>>>> imageArray, int inputSize) {
  img.Image image = img.Image.rgb(inputSize, inputSize);
  for (var x = 0; x < imageArray[0].length; x++) {
    for (var y = 0; y < imageArray[0][0].length; y++) {
      var r = (imageArray[0][x][y][0] * 255).toInt();
      var g = (imageArray[0][x][y][1] * 255).toInt();
      var b = (imageArray[0][x][y][2] * 255).toInt();
      image.setPixelRgba(x, y, r, g, b);
    }
  }
  return image;
}

MapEntry<String, double> getTopProbability(Map<String, double> labeledProb) {
  var pq = PriorityQueue<MapEntry<String, double>>(compare);
  pq.addAll(labeledProb.entries);

  return pq.first;
}

int compare(MapEntry<String, double> e1, MapEntry<String, double> e2) {
  if (e1.value > e2.value) {
    return -1;
  } else if (e1.value == e2.value) {
    return 0;
  } else {
    return 1;
  }
}
