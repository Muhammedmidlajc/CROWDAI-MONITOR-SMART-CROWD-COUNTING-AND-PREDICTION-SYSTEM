// ===============================
// AI Crowd Density Estimation
// ===============================

// File Upload Preview

const imageInput = document.getElementById("imageInput");

const uploadBox = document.querySelector(".upload-box");

if(imageInput){

imageInput.addEventListener("change",function(){

if(this.files.length>0){

uploadBox.querySelector("h3").innerHTML=this.files[0].name;

uploadBox.querySelector("p").innerHTML="Ready for Prediction";

}

});

}



// Drag & Drop

if(uploadBox){

uploadBox.addEventListener("dragover",e=>{

e.preventDefault();

uploadBox.style.borderColor="#00d4ff";

});

uploadBox.addEventListener("dragleave",()=>{

uploadBox.style.borderColor="rgba(255,255,255,.4)";

});

uploadBox.addEventListener("drop",e=>{

e.preventDefault();

imageInput.files=e.dataTransfer.files;

uploadBox.querySelector("h3").innerHTML=e.dataTransfer.files[0].name;

uploadBox.querySelector("p").innerHTML="Ready for Prediction";

});

}



// Loading Animation

const form=document.getElementById("uploadForm");

const btn=document.getElementById("predictBtn");

if(form){

form.addEventListener("submit",()=>{

document.getElementById("loadingScreen").style.display="flex";

btn.innerHTML="<i class='fas fa-spinner fa-spin'></i> AI Processing...";

btn.disabled=true;

});

}



// Animated Counter

const counter=document.getElementById("counter");

if(counter){

const target=parseInt(counter.innerText);

let count=0;

const speed=Math.max(1,Math.ceil(target/120));

const update=()=>{

count+=speed;

if(count>=target){

counter.innerText=target;

}else{

counter.innerText=count;

requestAnimationFrame(update);

}

}

update();

}



// Fade Images

const cards=document.querySelectorAll(".glass-card");

cards.forEach((card,index)=>{

card.style.opacity=0;

card.style.transform="translateY(30px)";

setTimeout(()=>{

card.style.transition=".8s";

card.style.opacity=1;

card.style.transform="translateY(0px)";

},index*150);

});



// Mouse Glow Effect

document.addEventListener("mousemove",(e)=>{

document.documentElement.style.setProperty(

"--mouseX",

e.clientX+"px"

);

document.documentElement.style.setProperty(

"--mouseY",

e.clientY+"px"

);

});

window.addEventListener("load", function () {

    const results = document.getElementById("results");

    if (results) {

        results.scrollIntoView({
            behavior: "smooth",
            block: "start"
        });

    }

});