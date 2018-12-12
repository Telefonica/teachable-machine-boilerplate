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
import { div } from "@tensorflow/tfjs";

// Number of classes to classify
const NUM_CLASSES = 4;
// Webcam Image size. Must be 227. 
const VIDEO_WIDTH = 700;
const VIDEO_HEIGHT = 400;
// K value for KNN
const TOPK = 10;

const availableAnimals = ['elefante', 'pajaro', 'mono', 'canguro', 'gallina', 'pinguino'];

function getRandom(arr, n) {
  var result = new Array(n),
      len = arr.length,
      taken = new Array(len);
  if (n > len)
      throw new RangeError("getRandom: more elements taken than available");
  while (n--) {
      var x = Math.floor(Math.random() * len);
      result[n] = arr[x in taken ? taken[x] : x];
      taken[x] = --len in taken ? taken[len] : len;
  }
  return result;
}

const images = getRandom(availableAnimals, 4);

class Main {
  constructor() {
    // Initiate variables
    this.infoTexts = [];
    this.teachableContainers = [];
    this.training = -1; // -1 when no class is being trained
    this.videoPlaying = false;
    this.trainedAnimals = [false, false, false, false]; // start training no Animals is being trained
    this.foundAnimals = [false, false, false, false]; // start playing no Animal is being found

    // Initiate deeplearn.js math and knn classifier objects
    this.bindPage();

    // Create video element that will contain the webcam image
    this.video = document.querySelector('#videoElement');

    // Create training buttons and info texts    
    for (let i = 0; i < NUM_CLASSES; i++) {
      
      // Get div for the animal
      const divAnimal = document.querySelector(`.animal-${i}`);

      // Get div for image
      const divImage = document.querySelector(`.animal-${i} .imageContainer`);

      const image = document.createElement('img');
      console.log(image.src);
      image.src = `images/${images[i]}.png`;
      divImage.appendChild(image);

      // Create div for text info
      const divText = document.querySelector(`.animal-${i} .textContainer`);      
      const infoText = document.createElement('p')
      infoText.innerText = 'Pendiente de Entrenar';
      divText.appendChild(infoText);
      
      this.infoTexts.push(infoText);
      
      // Create training buttons
      const divButton = document.querySelector(`.animal-${i} .buttonContainer`);
      const linkStart = document.createElement('a');
      linkStart.innerText = `Empezar`;
      linkStart.className = 'button';
      const linkStop = document.createElement('a');
      linkStop.innerText = `Detener`;
      linkStop.className = 'button';
      divButton.appendChild(linkStart);
      divButton.appendChild(linkStop);
      
      // Listen for mouse events when clicking the button
      linkStart.addEventListener('click', () => {
        console.log('trainging ===>', i);
        this.training = i;
        this.videoPlaying = true;
        this.trainedAnimals[i] = true;                 
      });

      linkStop.addEventListener('click', () => {
        console.log('stop training');
        this.training = -1;
        this.videoPlaying = false;
        cancelAnimationFrame(this.timer);
        if (this.checkAnimals(this.trainedAnimals)) {
          const modal = document.querySelector('.modal');
          modal.style.display = 'block';
          const headerButton = document.querySelector('.modal a');
          headerButton.innerHTML = 'Quiero jugar ya!';
          headerButton.style.display = 'block';          
        }        
      });      
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
    .catch((e) => {
      console.log('error getting video stream =>', e);
    });

    // Play or Train
    const start = document.querySelector('.modal a');
    start.addEventListener('click', () => {
      console.log('start ===>', start.innerHTML);
      if(start.innerHTML === 'Empezar a Entrenar') {
        console.log('entrenar ***********');
        this.start();
      } else if(start.innerHTML === 'Quiero jugar ya!') {
        console.log('Jugar **********');
        this.startGame();
      } else if(start.innerHTML === 'Eres una Fiera!!') {
        location.reload();
      }
    });
  }

  async bindPage() {
    this.knn = knnClassifier.create();
    this.mobilenet = await mobilenetModule.load();
  }

  start() {
    console.log('start ====>')
    if (this.timer) {
      this.stop();
    }
    this.video.play();
    this.timer = requestAnimationFrame(this.train.bind(this)); 
    
    const modal = document.querySelector('.modal');
    modal.style.display = 'none';
    const headerButton = document.querySelector('.modal a');
    headerButton.style.display = 'none';
  }

  stop() {
    this.video.pause();
    cancelAnimationFrame(this.timer);
  }

  checkAnimals(animals) {
    for (let i = 0; i < NUM_CLASSES; i++) {
      if (!animals[i]) {
        return false;
      }
    }
    return true;      
  }

  async train() {    
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
        // console.log('res.classIndex ===>', res);

        for (let i = 0; i < NUM_CLASSES; i++) {

          // The number of examples for each class
          const exampleCount = this.knn.getClassExampleCount();

          // Update info text
          if (exampleCount[i] > 0 && this.training != -1) {
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
    this.timer = requestAnimationFrame(this.train.bind(this));
  }

  startGame() {
    const modal = document.querySelector('.modal');
    modal.style.display = 'none';
    const headerButton = document.querySelector('.modal a');
    headerButton.style.display = 'none';

    this.videoPlaying = true;
    this.playGame();
  }

  async playGame() {
    if (this.videoPlaying) {
      const textContainers = document.getElementsByClassName('textContainer');
      // console.log('textContainers ===>', textContainers);
      for (let i = 0; i < textContainers.length; i ++) {
        textContainers[i].style.display = 'none';
      }

      const buttonContainers = document.getElementsByClassName('buttonContainer');    
      for (let i = 0; i < buttonContainers.length; i ++) {
        buttonContainers[i].style.display = 'none';
      }

      const divAnimal = document.getElementsByClassName('animal');
      const okMessage = document.querySelector('.ok-message');
      
      // Get image data from video element
      const image = tf.fromPixels(this.video);

      let logits;
      // 'conv_preds' is the logits activation of MobileNet.
      const infer = () => this.mobilenet.infer(image, 'conv_preds');

      const numClasses = this.knn.getNumClasses();
      if (numClasses > 0) {

        // If classes have been added run predict
        logits = infer();
        const res = await this.knn.predictClass(logits, TOPK);
        
        // let foundAnimal = false;

        for (let i = 0; i < NUM_CLASSES; i++) {
          const okAnimal = document.querySelector(`.animal-${i} .ok`);                
          // Make the predicted class bold
          if (res.classIndex == i) {
            console.log('ok i =>', i);

            divAnimal[i].style.opacity = '0.3';                        
            divAnimal[i].style.filter = 'alpha(opacity=30)';
            console.log('div Animal [i] ===>', divAnimal[i]);
            okAnimal.style.display = 'block';

            console.log('ok-image ==>', `images/ok-${images[i]}.png`);
            okMessage.innerHTML = `Pareces un ${images[i]}`;
            if (images[i] == 'gallina') {
              okMessage.innerHTML = `Pareces una ${images[i]}`;
            }
            this.foundAnimals[i] = true;
            console.log('found Animals ===>', this.foundAnimals)
            if(this.checkAnimals(this.foundAnimals)) {
              console.log('it should stop the game')
              this.stopGame();
            }

            // Hide video and display animal ---> does not look right
            // this.video.style.display = 'none';
            // okAnimalImg.src = `images/ok-${images[i]}.png`;
            // okAnimal.style.display = 'block';
            // foundAnimal = true;
          } else {
            // no quiar opacidad una vez encontrado    
            // divAnimal[i].style.opacity = '1';                  
            // divAnimal[i].style.filter = 'alpha(opacity=1)';            

            // If animal not found return video to show and hide image.

            // if (!foundAnimal) {
            //   okAnimal.style.display = 'none';
            //   okAnimalImg.src = ``;
            //   this.video.style.display = 'block';
            // }
          }
        }
      }

      // Dispose image when done
      image.dispose();
      if (logits != null) {
        logits.dispose();
      }
    }
    this.timer = requestAnimationFrame(this.playGame.bind(this));
  }

  stopGame() {
    this.stop();
    this.videoPlaying = false;
    const modal = document.querySelector('.modal');
    modal.style.display = 'block';
    const headerButton = document.querySelector('.modal a');
    headerButton.innerHTML = 'Eres una Fiera!!';
    headerButton.style.display = 'block';    
  }
}

window.addEventListener('load', () => new Main());