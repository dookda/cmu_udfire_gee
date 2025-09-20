// get data from join query
app.get('/api/no2_urban/:date', async (req, res) => {
    try {
        const { date } = req.params;


        const sql = `SELECT a.id, ST_AsGeoJSON(geom) as geom, b.no2
                FROM  public.urban30 AS a
                JOIN  public.co_7day_forecast_urban AS b
                      ON a.id::text = b.id
                WHERE  b.date ='${date}'`

        console.log(sql);
        const result = await pool.query(sql);

        res.json(result.rows);
    } catch (err) {
        console.error(err);
        res.status(500).json({ error: 'Internal Server Error' });
    }
});

SELECT * FROM co_7day_forecast_urban