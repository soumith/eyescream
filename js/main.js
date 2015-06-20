nn = require('nn');

function getRandom(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

screamer = {};

screamer.C = document.getElementById("screamer");
/* make the grid nicely aligned to 64-pixel tiles */
screamer.C.height = (window.innerHeight + 64) - (window.innerHeight % 64);
screamer.C.width = (window.innerWidth + 64) - (window.innerWidth % 64);
screamer.height = screamer.C.height;
screamer.width = screamer.C.width;
screamer.nH = screamer.height / 64;
screamer.nW = screamer.width / 64;
console.log(screamer.nH, screamer.nW);

screamer.startH = function(grididh) {
    return grididh * 64;
}

screamer.startW = function(grididw) {
    return grididw * 64;
}

/* just assuming that the canvas element is present. bad programming ftw! */
var ctx = screamer.C.getContext("2d");
screamer.B = ctx.createImageData(64, 64); /* reusable buffer for one tile */


screamer.random = function () {
    var h = screamer.startH(getRandom(0, screamer.nH-1))
    var w = screamer.startW(getRandom(0, screamer.nW-1))
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

for (i=0; i < 10000; i++) {
    screamer.fake();
    screamer.random();
}

/* load network */



screamer.hallucinate = function() {
    
}
