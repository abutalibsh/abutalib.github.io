
## My Experience Running Custom Pytorch Models On Java

Even in software engineering, our daily job can be boring. But every once in a while, there comes a new problem that get people frustrated since they weren’t prepared for. I love those. I think I’ve chosen programming because of them – when I have the chance.

It was a seemingly simple task. Our company has data scientists and good ones. They create models that developers have to use. The task is to simply use a pre-trained model against images and detect whether there were people or vehicles in this image or not. Easy enough, isn’t it?

Here is what I didn’t know. Data scientists are not developers. They don’t think in terms of user experience or designing applications. They may use python and some programming libraries to carry out their tasks, but not in a way that is meant for an end-user consumption - let alone a developer consumption.

Now, I develop in Java, but I’ve dappled in python before, so it wasn’t an issue. However, we had a few challenges with running python in our application:

-       Java is multithreaded: python has multi-processing but isn’t multithreaded. So, it will load the model (around 45MB) to memory with each new process it creates. On the other hand, Java will load a single model and many threads will reference it.

-       Backlog: we had a 2 years old backlog worth of images that we needed to run inference on. We planned to use jobs to process them.

-       Java is the tech stack: so, using python is not going to be easily maintainable. Plus, we need to find a way to use python in our production, which was not possible.

With that in mind, Java looked like the only option to go for.

The problem is that I didn’t know anything about artificial intelligence. I mean, even when I google or ask chatgpt, I didn’t even know what words to use to describe my problem. So, I had to learn quickly whatever terms necessary to start working on the task at hand.

My first break was when I found an AI library developed by Amazon called DJL (Deep Java Library). Amazon, did a great job of creating a library that I can use to run inference on the images in order to detect objects inside them.

The problem was that there wasn’t much written about DJL, except for a book that was too technical and deep for me to read. So, I had to settle for some code examples. Here is the typical boilerplate that should run the whole thing smoothly.

```java
public final class ObjectDetection {

private static final Logger logger = LoggerFactory.getLogger(ObjectDetection.class);

private ObjectDetection() {}

public static void main(String[] args) throws IOException, ModelException, TranslateException {

DetectedObjects detection = predict();

logger.info("{}", detection);

}

public static DetectedObjects predict() throws IOException, ModelException, TranslateException {

Path imageFile = Paths.get("src/test/resources/dog_bike_car.jpg");

Image img = ImageFactory.getInstance().fromFile(imageFile);

Criteria<Image, DetectedObjects> criteria =

Criteria.builder()

.optApplication(Application.CV.OBJECT_DETECTION)

.setTypes(Image.class, DetectedObjects.class)

.optArgument("threshold", 0.5)

.optEngine("PyTorch")

.optProgress(new ProgressBar())

.build();

try (ZooModel<Image, DetectedObjects> model = criteria.loadModel();

Predictor<Image, DetectedObjects> predictor = model.newPredictor()) {

DetectedObjects detection = predictor.predict(img);

saveBoundingBoxImage(img, detection);

return detection;

}

}

private static void saveBoundingBoxImage(Image img, DetectedObjects detection)

throws IOException {

Path outputDir = Paths.get("build/output");

Files.createDirectories(outputDir);

img.drawBoundingBoxes(detection);

Path imagePath = outputDir.resolve("detected-dog_bike_car.png");

// OpenJDK can't save jpg with alpha channel

img.save(Files.newOutputStream(imagePath), "png");

logger.info("Detected objects image has been saved in: {}", imagePath);

}

}
```

If you can't find the model name anywhere in this example code, then I sympathize with you. You see, the actual model loaded depends on the default model provided by DJL for the given application and engine. Without specifying a specific model, DJL will use its default object detection model for PyTorch.

The second part of the puzzle was understanding what (.pt) files stood for? How do I open them? How can I load them to DJL? I guessed they stood for “pre-trained” model. Silly me.

It turned out they stood for pytorch, which is one of the libraries used to create pre-trained models, and this was their format. Now, DJL only supports pytorch in some serialized format that I didn’t quite understand. I tried converting the models myself, but hey, I am not a wizard nor a data scientist. But I did find out that there was a format accepted by DJL called ONNX (Open Neural Network Exchange). So, I ran this by the data scientist, and he was glad to help. He indeed converted the pytorch models to ONNX, and now I had a model that I could use in DJL. So naturally, I changed the following line and that's it. Happy ending, right?

```java

.optEngine("OnnxRuntime")

.optProgress(new ProgressBar())

.build();
```

When I ran the models using ONNX, they didn’t run. I had configured the engine to use ONNX as a runtime, but it still didn’t succeed. Nothing at this point was helping me, neither chat-gpt 4, nor google. Nothing. I was in uncharted territory with a deadline over my head to meet. But, then I started debugging the code in Intellij, and I found that it was looking for a pytorch engine as well. So, I tried adding a pytorch engine maven dependency, and it suddenly didn’t give me the same error.

```
<dependency>  
    <groupId>ai.djl.pytorch</groupId>  
    <artifactId>pytorch-engine</artifactId>  
</dependency>
```

As it turned out, ONNX runtime couldn’t cover every bit of functionality that pytorch engine could do, so DJL used a “hybrid engine” approach, were it would delegate some of the pytroch functions to its native libraries (more on those babies later) without changing my code.

I started going back and forth with the data scientist, and he would throw me a bone once in a while, as far as AI goes, which was really helpful. But I still needed to hack DJL and understand how it works. But at least I got it to run.

The next problem is that it didn’t give correct results. I used the same sample images that I used with the python sample code that will run inference, and it was giving me incorrect results. I went back to the data scientist, who explained to me NMS, and I discovered that there was something called YOLO translator in Java, which I didn’t know about. After tinkering for a while I found 2 things:

1.     Relative YOLO V8 Translator: I discovered I needed a different translator because I needed to do some normalization. Normalization means I needed to recalculate the bounding boxes (the little square boxes around faces) as per the original image's coordinates. Why? Because this is a custom model and it accepts the image with a certain size, and produces them accordingly.

```java
private DetectedObjects processFromBoxOutput(NDList list) {  
        float[] flattened = list.get(0).transpose().toFloatArray();  
  
        ArrayList<IntermediateResult> intermediateResults = new ArrayList<>();  
        int sizeClasses = classes.size();  
        int stride = 4 + sizeClasses;  
        int size = flattened.length / stride;  
        for (int i = 0; i < size; i++) {  
            int indexBase = i * stride;  
            float maxClass = 0;  
            int maxIndex = 0;  
            for (int c = 0; c < sizeClasses; c++) {  
                if (flattened[indexBase + c + 4] > maxClass) {  
                    maxClass = flattened[indexBase + c + 4];  
                    maxIndex = c;  
                }  
            }  
            float score = maxClass;  
            if (score > threshold) {                  
                float xPos = flattened[indexBase];  
                float yPos = flattened[indexBase + 1];  
                float w = flattened[indexBase + 2];  
                float h = flattened[indexBase + 3];  
//                System.err.println("Class : " + classes.get(maxIndex) + ", xpos: " + xPos + ", ypos: " + yPos + ", w: " + w + ", h: " + h);  
                Rectangle rect =  
                        new Rectangle(Math.max(0, xPos - w / 2), Math.max(0, yPos - h / 2), w, h);  
                intermediateResults.add(  
                        new IntermediateResult(classes.get(maxIndex), score, maxIndex, rect));  
            }  
        }  
        DetectedObjects output = nms(intermediateResults);  
//        if(1==1) return output;  
        List<String> classList = new ArrayList<>();  
        List<Double> probList = new ArrayList<>();  
        List<BoundingBox> rectList = new ArrayList<>();  
  
        final List<DetectedObjects.DetectedObject> items = output.items();  
        items.forEach(item -> {  
            classList.add(item.getClassName());  
            probList.add(item.getProbability());  
  
            Rectangle b = item.getBoundingBox().getBounds();  
            Rectangle newBox = new Rectangle(b.getX() / imageWidth,  
                    b.getY() / imageHeight,  
                    b.getWidth() / imageWidth,  
                    b.getHeight() / imageHeight);  
  
            rectList.add(newBox);  
        });  
        return new DetectedObjects(classList, probList, rectList);  
    }
```

2.     Model custom image resizing: you see, since this wasn’t you run-of-the-mill model that is created for DJL library, it had a pre-condition, which is that the image needed to be transformed to 320x320 px before inference was run on it. I found out where I need to configure programmatically. Therefore I needed placeholders for handling specific output formats and decoding bounding boxes.


```java
private static YoloV8RelativeTranslator detectionTranslator=YoloV8RelativeTranslator.builder()  
       .optThreshold(.5f)  
       .optNmsThreshold(.5f)  
       .optOutputType(YoloOutputType.BOX)  
       .optImageWidth(320)  
       .optImageHeight(320)  
       .optSynset(List.of("Car","Face"))  
       .setPipeline(new Pipeline()  
        .add(new Resize(320, 320)) // make height to 320 width to 320  
        .add(new ToTensor())  
        )  
       .build();

Criteria<Image, DetectedObjects> criteria =  
       Criteria.builder()  
       .setTypes(Image.class, DetectedObjects.class)  
       .optApplication(Application.CV.OBJECT_DETECTION)  
       .optArgument("width", 320)  
       .optArgument("height", 320)  
       .optTranslator(detectionTranslator)  
       .optArgument("softmax", true)  
       .optModelPath(qualityModel)               
       .optEngine("OnnxRuntime")  
       .build();  
return criteria.loadModel();
```


After that, everything fell into its place, and I started running inference on the project without an issue. I developed a rest controller that would take an image and return inference results.

##### Openshift, DJL and PyTorch Native Libraries:

Developing locally was one thing, but there was another gotcha hidden in the deployment. After the build from the pipeline was successful, we discovered our project was failing. After debugging, we found that DJL was trying to download some pytorch native libraries from the internet (remember those babies?), and our DEV environment naturally didn’t allow it!

It turned out that DJL downloads some native libraries from the internet and caches them locally, but only the first time you run inference. I had forgot this happened at all. Also, it was not downloading any libraries willy-nilly, but rather it downloaded specific files depending on the OS the project was run from (so much for build once and run everywhere Java heh?). Amazon had to resort to this behavior because the size of those libraries were huge (roughly 1GB), and they couldn’t just shove them with every project, so they had to develop this workaround. But this meant that the DLL files downloaded for my windows (.so files for Linux), couldn’t be used in our custom alpine Docker image. To make matters worse, the custom image didn’t allow downloading anything from Amazon repository since it was a random website and not a Maven repository. Eventually the Operation team solved the problem by whitelisting the website, which enabled the download. And that was our happy ending.

It was definitely an interesting task, which was remarkable and taught me a lot of things. A huge credit goes to the data scientist who helped me Mr. Abdulaziz Almojil, who was very supportive, and I couldn’t do this without him being on my side every step of the way. I hope this can help other Java developers use the information here to discover the capabilities of DJL library and find my journal on it useful.