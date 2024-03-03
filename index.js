// 导入所需的库和模块
const fs = require('fs');
const path = require('path');
const { createCanvas, loadImage } = require('canvas');
const faceapi = require('face-api.js');

// 设置模型路径
const modelsPath = path.join(__dirname, '/models');
faceapi.nets.ssdMobilenetv1.loadFromDisk(modelsPath);
faceapi.nets.faceRecognitionNet.loadFromDisk(modelsPath);

// 创建图像识别工具类
class ImageRecognitionTool {
    constructor() {
        this.modelsLoaded = false;
        this.canvas = createCanvas(800, 600);
    }

    // 加载模型
    async loadModels() {
        await faceapi.nets.ssdMobilenetv1.loadFromDisk(modelsPath);
        await faceapi.nets.faceRecognitionNet.loadFromDisk(modelsPath);
        this.modelsLoaded = true;
    }

    // 识别对象
    async recognizeObjects(imagePath) {
        if (!this.modelsLoaded) {
            throw new Error("Models are not loaded.");
        }

        const image = await loadImage(imagePath);
        const detections = await faceapi.detectAllFaces(image).withFaceLandmarks().withFaceDescriptors();
        
        // 识别人脸
        if (detections.length > 0) {
            console.log("Faces detected:");
            detections.forEach((detection, index) => {
                console.log(`Face ${index + 1}: ${detection.descriptor}`);
            });
        } else {
            console.log("No faces detected.");
        }
    }

    // 分类图像
    async classifyImage(imagePath) {
        if (!this.modelsLoaded) {
            throw new Error("Models are not loaded.");
        }

        const image = await loadImage(imagePath);
        const detections = await faceapi.detectAllFaces(image).withFaceLandmarks().withFaceDescriptors();
        
        // 分类对象
        if (detections.length > 0) {
            console.log("Image contains faces.");
        } else {
            console.log("Image does not contain faces.");
        }
    }
}

// 示例用法
async function main() {
    const recognitionTool = new ImageRecognitionTool();
    await recognitionTool.loadModels();

    const imagePath = path.join(__dirname, '/images/example.jpg');

    console.log("Recognizing objects in the image...");
    await recognitionTool.recognizeObjects(imagePath);

    console.log("\nClassifying the image...");
    await recognitionTool.classifyImage(imagePath);
}

main().catch(error => {
    console.error("An error occurred:", error);
});
