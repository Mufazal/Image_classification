import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:imageclassification/classifier.dart';
import 'package:integration_test/integration_test.dart';
import 'package:tflite_flutter_helper/tflite_flutter_helper.dart';

import 'classifier_test_helper.dart';

const sampleFileName = 'assets/lion.jpg';
const labelFileName = 'assets/labels.txt';

const model_float = 'mobilenet_v1_1.0_224.tflite';
const model_quant = 'mobilenet_v1_1.0_224_quant.tflite';

//flutter driver --driver='test_driver/image_classification_e2e_test.dart' test/image_classification_e2e.dart
void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('inference', () {
    img.Image testImage;

    setUp(() async {
      ByteData imageFile = await rootBundle.load(sampleFileName);
      testImage = img.decodeImage(imageFile.buffer.asUint8List());
    });

    group('float', () {
      Classifier classifier;

      setUpAll(() {
        classifier = ClassifierFloatTest();
      });

      test('run', () {
        Category prediction = classifier.predict(testImage);
        expect(prediction.label, "lion");
      });

      tearDownAll(() {
        classifier.close();
      });
    });
    group('quant', () {
      ClassiferTest classifier;

      setUpAll(() {
        classifier = ClassifierQuantTest();
      });

      test('run', () {
        Category prediction = classifier.predict(testImage);
        expect(prediction.label, "lion");
      });

      tearDownAll(() {
        classifier.close();
      });
    });
  });
}
