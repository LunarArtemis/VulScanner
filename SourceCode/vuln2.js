const express = require('express');
const fs = require('fs');
const path = require('path');

const app = express();
const port = 3000;

app.get('/file', (req, res) => {
    const fileName = req.query.file;
    const filePath = path.join(__dirname, fileName);
    fs.readFile(filePath, 'utf8', (err, data) => {
        if (err) {
            return res.status(404).send('File not found!');
        }
        res.send(data);
    });
});

app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});
