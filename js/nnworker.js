importScripts('nn.js');

debugprints = false;

var ndarray = require('ndarray');
var nn = require('nn');

function getRandom(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

function print() {
    if (console && console.log && debugprints) {
	console.log.apply(console, arguments)
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

loadNet('../models/8x8.mpac', 0);
loadNet('../models/8x14.mpac', 1);
loadNet('../models/14x28.mpac', 2);


onmessage = function(e) {
    if (network.length < 3) {
	postMessage([]);
	return
    }
    var noise = [];
    noise[0] = nn.utils.noise1d(100);
    noise[1] = nn.utils.noise3d(1,14,14);
    noise[2] = nn.utils.noise3d(1,28,28);

    var class_num = getRandom(0, 9);
    var one_hot = [];
    var one_hot_data = new Float32Array(10);
    for (i=0; i < 10; i++)
	one_hot_data[i] = 0;
    one_hot_data[class_num] = 1
    one_hot[0] = new ndarray(one_hot_data, [10])
    one_hot[1] = new ndarray(one_hot_data, [10, 1, 1])
    one_hot[2] = new ndarray(one_hot_data, [10, 1, 1])

    var o = []; // outputs of each scale
    var t1 = performance.now();
    o[0] = network[0].forward([noise[0], one_hot[0]])
    var t2 = performance.now();
    print('network 1 time: ', t2 - t1)
    o[0] = nn.utils.upscaleBilinear(o[0], 14, 14);
    var t1 = performance.now();
    o[1] = network[1].forward([noise[1], one_hot[1], o[0]])
    var t2 = performance.now();
    print('network 2 time: ', t2 - t1)
    for (i=0; i < o[1].data.length; i++)
	o[1].data[i] = o[1].data[i] + o[0].data[i];
    o[1] = nn.utils.upscaleBilinear(o[1], 28, 28);
    var t1 = performance.now();
    o[2] = network[2].forward([noise[2], one_hot[2], o[1]])
    var t2 = performance.now();
    print('network 3 time: ', t2 - t1)
    for (i=0; i < o[2].data.length; i++)
	o[2].data[i] = o[2].data[i] + o[1].data[i];
    nn.utils.minMaxNormalize(o[2]);
    o[2] = nn.utils.upscaleBilinear(o[2], 64, 64);

    postMessage(o[2]);
}
