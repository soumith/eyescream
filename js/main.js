nn = require('nn');

function getRandom(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

function print() {
    if (console && console.log) {
	console.log.apply(console, arguments)
    }
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

for (i=0; i < screamer.nH; i++) {
    for (j=0; j < screamer.nW; j++) {
	screamer.fake();
	screamer.draw(i, j);
    }
}

/* load network from server */
var network = [];

function loadNet(url, idx) {
    var sock = new XMLHttpRequest();
    sock.open("GET", url, true);
    sock.responseType = "arraybuffer";
    sock.onload = function (oEvent) {
	var arrayBuffer = sock.response;
	if (arrayBuffer) {
	    var byteArray = new Uint8Array(arrayBuffer);
	    network[idx] = nn.loadFromMsgPack(byteArray);
	    print('succesfully loaded : ', url, idx)
	}
    };
    sock.send(null);
}

loadNet('models/8x8.mpac', 0);
loadNet('models/8x14.mpac', 1);
loadNet('models/14x28.mpac', 2);

screamer.hallucinate = function() {
    /* while we are waiting for the networks to download, show pregenerated images */
    if (network.length < 3) {
	
    } else { /* all models loaded, do live-gen */
	
    }
}
