let data = [];

function setup() {
    createCanvas(windowWidth, windowHeight - 100);
    fetch('/data')
        .then(response => response.json())
        .then(json => {
            data = json;
            drawPoints();
        });
}

function drawPoints() {
    background(255);
    textAlign(CENTER, CENTER);
    fill(0);
    data.forEach(item => {
        let x = map(item.x, -2, 2, 50, width - 50);
        let y = map(item.y, -2, 2, 50, height - 50);
        ellipse(x, y, 6, 6);
        textSize(12);
        text(item.word, x, y - 10);
    });
}
