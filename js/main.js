var nn = require('nn');
var ndarray = require('ndarray');

debugprints = false;

function getRandom(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

function print() {
    if (console && console.log && debugprints) {
	console.log.apply(console, arguments)
    }
}

screamer = {};

screamer.C = document.getElementById("screamer");
/* make the grid nicely aligned to 64-pixel tiles */
screamer.sz = 64
screamer.C.height = (window.innerHeight + screamer.sz) - (window.innerHeight % screamer.sz);
screamer.C.width = (window.innerWidth + screamer.sz) - (window.innerWidth % screamer.sz);
screamer.height = screamer.C.height;
screamer.width = screamer.C.width;
screamer.nH = screamer.height / screamer.sz;
screamer.nW = screamer.width / screamer.sz;

screamer.startH = function(grididh) {
    return grididh * screamer.sz;
}

screamer.startW = function(grididw) {
    return grididw * screamer.sz;
}

/* just assuming that the canvas element is present. bad programming ftw! */
var ctx = screamer.C.getContext("2d");
screamer.B = ctx.createImageData(screamer.sz, screamer.sz); /* reusable buffer for one tile */


screamer.random = function () {
    var h = screamer.startH(getRandom(0, screamer.nH-1))
    var w = screamer.startW(getRandom(0, screamer.nW-1))
    ctx.putImageData(screamer.B, w, h);
}

screamer.draw = function (i, j) {
    var h = screamer.startH(i)
    var w = screamer.startW(j)
    ctx.putImageData(screamer.B, w, h);
}

screamer.fake = function() {
    var r = getRandom(0, 255);
    var g = getRandom(0, 255);
    var b = getRandom(0, 255);
    for (var i = 0; i < screamer.B.data.length; i += 4) {
	screamer.B.data[i+0] = r
	screamer.B.data[i+1] = g
	screamer.B.data[i+2] = b
	screamer.B.data[i+3] = 255;
    }
}

screamer.fillArray = function(arr) {
    var iH = arr.shape[1];
    var iW = arr.shape[2];
    arr.transpose(0, 2);
    var cH = 0;
    var cW = 0;
    for (var i=0; i < screamer.sz && i < iH; i++) {
	for (var j=0; j < screamer.sz && j < iW; j++) {
	    var idx = (i * screamer.sz + j) * 4
	    screamer.B.data[idx+0] = arr.get(0, i, j);
	    screamer.B.data[idx+1] = arr.get(1, i, j);
	    screamer.B.data[idx+2] = arr.get(2, i, j);
	    screamer.B.data[idx+3] = 255;
	}
    }
}

for (i=0; i < screamer.nH; i++) {
    for (j=0; j < screamer.nW; j++) {
	screamer.fake();
	screamer.draw(i, j);
    }
}

var showPreGen = function() {
    var img = new Image();
    img.onload = function () {
	var h = screamer.startH(getRandom(0, screamer.nH-1))
	var w = screamer.startW(getRandom(0, screamer.nW-1))
	ctx.drawImage(img, w, h, screamer.sz, screamer.sz);
    }
    img.src = "pregen/" + getRandom(1,6000) + ".png";
    print('missed worker, showing pregen');
}

var workerDraw = function(e) {
    var out = e.data;
    if (out.length == 0) {
	/* while we are waiting for the networks to download, show pregenerated images */
	showPreGen();
    } else {
	var o2 = ndarray(out.data, out.shape);
	screamer.fillArray(o2);
	screamer.random();
    }
}

var workerCB = [];
workerCB[0] = function(e) { working[0] = false; workerDraw(e); }
workerCB[1] = function(e) { working[1] = false; workerDraw(e); }
workerCB[2] = function(e) { working[2] = false; workerDraw(e); }
workerCB[3] = function(e) { working[3] = false; workerDraw(e); }

var worker = []
var working = []
for (i=0; i < 4; i++) {
    worker[i] = new Worker("js/nnworker.js");
    worker[i].onmessage = workerCB[i];
    working[i] = false;
}

screamer.hallucinate = function() {
    for (i=0; i < worker.length; i++) {
	if (working[i] == false) {
	    worker[i].postMessage([]);
	    working[i] = true;
	    return;
	}
    }
    showPreGen(); /* if all workers are busy before timeout, show pregen images */
}

screamer.hallucinate();
setInterval(screamer.hallucinate, 300);



