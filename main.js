// setup canvas

const canvas = document.querySelector('canvas');
const ctx = canvas.getContext('2d');

const width = canvas.width = window.innerWidth;
const height = canvas.height = window.innerHeight;

// function to generate random number

function random(min, max) {
    const num = Math.floor(Math.random() * (max - min + 1)) + min;
    return num;
}

function Ball(x, y, velX, velY, color, size) {
    this.x = x;
    this.y = y;
    this.velX = velX;
    this.velY = velY;
    this.color = color;
    this.size = size;
    this.gravity = 2;
    this.friction = 0;
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
        this.gravity = 0;
        this.friction = 10;
    } else {
        this.gravity = 2;
        this.friction = 0;
    }

    if ((this.y) <= 0) {
        this.velY = -(this.velY);
    }
    /*if (this.velX > 0) {
        this.velX = Math.max(this.velX - this.friction * time, 0)
    } else if (this.velX > 0) {
        this.velX = Math.min(this.velX + this.friction * time, 0)
    }*/

    this.velY += this.gravity * time
    this.x += this.velX * time;
    this.y += this.velY * time;
}


let balls = [];

while (balls.length < 25) {
    let size = random(10,20);
    let ball = new Ball(
        // ball position always drawn at least one ball width
        // away from the edge of the canvas, to avoid drawing errors
        random(0 + size,width - size),
        random(0 + size,height - size),
        random(-4,4),
        random(-4,4),
        'rgb(' + random(0,255) + ',' + random(0,255) + ',' + random(0,255) +')',
        size
    );

    balls.push(ball);
}

const time = {
    start: performance.now()
}

function loop() {
    ctx.fillStyle = 'rgba(0, 0, 0, 0.25)';
    ctx.fillRect(0, 0, width, height);

    for (let i = 0; i < balls.length; i++) {
        balls[i].draw();
        balls[i].collisionDetect();
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
                balls[j].color = this.color = 'rgb(' + random(0, 255) + ',' + random(0, 255) + ',' + random(0, 255) +')';
            }
        }
    }
}

loop()