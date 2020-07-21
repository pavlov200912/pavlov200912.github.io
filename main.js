// setup canvas

const canvas = document.querySelector('canvas');
const ctx = canvas.getContext('2d');

const width = canvas.width = window.innerWidth;
const height = canvas.height = window.innerHeight;

// function to generate random number

function random(min, max) {
    const num = Math.random() * (max - min + 1) + min;
    return num;
}

function Ball(x, y, velX, velY, color, size) {
    this.x = x;
    this.y = y;
    this.velX = velX / 5;
    this.velY = velY / 5;
    this.color = color;
    this.size = size
}

Ball.prototype.draw = function() {
    ctx.beginPath();
    ctx.fillStyle = this.color;
    ctx.fillRect(this.x, this.y, this.size, this.size);
    ctx.fill();
}

Ball.prototype.update = function (time) {
    time = time / 20
    if ((this.x + this.size) >= width) {
        this.velX = -(this.velX);
    }

    if ((this.x) <= 0) {
        this.velX = -(this.velX);
    }

    if ((this.y + this.size) >= height) {
        this.velY = -(this.velY);
    }

    if ((this.y) <= 0) {
        this.velY = -(this.velY);
    }

    this.x += this.velX;
    this.y += this.velY;
}


let balls = [];

while (balls.length < 50) {
    let size = 25;
    let ball = new Ball(
        // ball position always drawn at least one ball width
        // away from the edge of the canvas, to avoid drawing errors
        random(size,width - size),
        random(size,height - size),
        random(-0.5,0.5),
        random(-0.5,0.5),
        'rgb(129, 156, 169)',
        size
    );

    balls.push(ball);
}

const time = {
    start: performance.now()
}

function loop() {
    ctx.fillStyle = 'rgb(41, 67, 78)';
    ctx.fillRect(0, 0, width, height);

    for (let i = 0; i < balls.length; i++) {
        balls[i].draw();
       // balls[i].collisionDetect();
        balls[i].update(performance.now() - time.start);
    }
    time.start = performance.now()
    requestAnimationFrame(loop);
}

Ball.prototype.collisionDetect = function() {
    for (let j = 0; j < balls.length; j++) {
        if (!(this === balls[j])) {
            const dx = this.x - balls[j].x;
            const dy = this.y - balls[j].y;
            const distance = Math.sqrt(dx * dx + dy * dy);

            if (distance < this.size + balls[j].size) {
             // ???
            }
        }
    }
}

loop()