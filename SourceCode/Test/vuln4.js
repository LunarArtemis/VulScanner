const moment = require('moment');
const app = express();
const port = 1337;

app.get('/time', (req, res) => {
    const locale = req.query.locale || 'en'; 

    // CVE-2022-24785 triggers at the following line in locate() function
    // locale() function passes the first parameter to require() without any sanitisation, this makes it easier to perform a path traversal attack
    const currentTime = moment().locale(locale).format('LLLL');

    res.send(`Current time (${locale}): ${currentTime}`);
});

app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
});