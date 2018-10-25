// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import "@babel/polyfill";
import * as mobilenetModule from '@tensorflow-models/mobilenet';
import * as tf from '@tensorflow/tfjs';
import * as knnClassifier from '@tensorflow-models/knn-classifier';

// Number of classes to classify
const NUM_CLASSES = 4;
// Webcam Image size. Must be 227. 
const VIDEO_WIDTH = 500;
const VIDEO_HEIGHT = 250;
// K value for KNN
const TOPK = 10;

const images = ['elefante', 'pajaro', 'mono', 'canguro'];

class Main {
  constructor() {
    // Initiate variables
    this.infoTexts = [];
    this.teachableContainers = [];
    this.training = -1; // -1 when no class is being trained
    this.videoPlaying = false;

    // Initiate deeplearn.js math and knn classifier objects
    this.bindPage();

    const container = document.createElement('div');
    container.className = 'container-fluid'
    document.body.appendChild(container);

    const videoDiv = document.createElement('div');
    videoDiv.className = 'row justify-content-center video-container';
    container.appendChild(videoDiv);

    // Create video element that will contain the webcam image
    this.video = document.createElement('video');
    this.video.setAttribute('autoplay', '');
    this.video.setAttribute('playsinline', '');

    // Add video element to DOM
    videoDiv.appendChild(this.video);    

    let rowDiv;
    // Create training buttons and info texts    
    for (let i = 0; i < NUM_CLASSES; i++) {

      if (i%2 === 0) {
        // Add div for row
        rowDiv = document.createElement('div');
        rowDiv.className = 'row justify-content-between fila'
        container.appendChild(rowDiv); 
      }   

      // Add div for class
      const div = document.createElement('div');
      rowDiv.appendChild(div);
      div.style.marginBottom = '10px';
      div.className = 'teachable-container col-3';

      
      // Add images
      const divImage = document.createElement('div');
      divImage.className = 'row justify-content-center align-items-center image-container'
      div.appendChild(divImage);
      const image = document.createElement('img');
      console.log(image.src);
      image.src = `images/${images[i]}.png`;      
      divImage.appendChild(image);
      
      // Create training button
      const divButton = document.createElement('div');
      divButton.className = 'row justify-content-center button-container'
      div.appendChild(divButton);
      const button = document.createElement('button')
      button.innerText = `Entrena al ${images[i]}`;
      button.className = 'btn btn-outline-primary'
      divButton.appendChild(button);
      
      // Listen for mouse events when clicking the button
      button.addEventListener('mousedown', () => this.training = i);
      button.addEventListener('mouseup', () => this.training = -1);
      
      // Create info text
      const divSpam = document.createElement('div');
      divSpam.className = 'row justify-content-center'
      div.appendChild(divSpam);
      
      const infoText = document.createElement('span')
      infoText.innerText = 'Pendiente de Entrenar';
      divSpam.appendChild(infoText);
      
      this.infoTexts.push(infoText);
      this.teachableContainers.push(div);
    }


    // Setup webcam
    navigator.mediaDevices.getUserMedia({ video: true, audio: false })
      .then((stream) => {
        this.video.srcObject = stream;
        this.video.width = VIDEO_WIDTH;
        this.video.height = VIDEO_HEIGHT;

        this.video.addEventListener('playing', () => this.videoPlaying = true);
        this.video.addEventListener('paused', () => this.videoPlaying = false);
      })
  }

  async bindPage() {
    this.knn = knnClassifier.create();
    this.mobilenet = await mobilenetModule.load();

    this.start();
  }

  start() {
    if (this.timer) {
      this.stop();
    }
    this.video.play();
    this.timer = requestAnimationFrame(this.animate.bind(this));
  }

  stop() {
    this.video.pause();
    cancelAnimationFrame(this.timer);
  }

  async animate() {
    if (this.videoPlaying) {
      // Get image data from video element
      const image = tf.fromPixels(this.video);

      let logits;
      // 'conv_preds' is the logits activation of MobileNet.
      const infer = () => this.mobilenet.infer(image, 'conv_preds');

      // Train class if one of the buttons is held down
      if (this.training != -1) {
        logits = infer();

        // Add current image to classifier
        this.knn.addExample(logits, this.training)
      }

      const numClasses = this.knn.getNumClasses();
      if (numClasses > 0) {

        // If classes have been added run predict
        logits = infer();
        const res = await this.knn.predictClass(logits, TOPK);

        for (let i = 0; i < NUM_CLASSES; i++) {

          // The number of examples for each class
          const exampleCount = this.knn.getClassExampleCount();

          // Make the predicted class bold
          if (res.classIndex == i) {
            this.infoTexts[i].style.fontWeight = 'bold';
            this.teachableContainers[i].style.backgroundColor = '#FF0066';                        
          } else {
            this.infoTexts[i].style.fontWeight = 'normal';
            this.teachableContainers[i].style.backgroundColor = '#ffffff'; 
          }

          // Update info text
          if (exampleCount[i] > 0) {
            this.infoTexts[i].innerText = ` ${exampleCount[i]} examples - ${res.confidences[i] * 100}%`
          }
        }
      }

      // Dispose image when done
      image.dispose();
      if (logits != null) {
        logits.dispose();
      }
    }
    this.timer = requestAnimationFrame(this.animate.bind(this));
  }
}

window.addEventListener('load', () => new Main());