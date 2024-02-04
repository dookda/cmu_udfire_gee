exports.temporalCollection = function (collection, start, count, interval, units) {
    // Create a sequence of numbers, one for each time interval.
    var sequence = ee.List.sequence(0, ee.Number(count).subtract(1));

    var originalStartDate = ee.Date(start);

    return ee.ImageCollection(sequence.map(function (i) {
        // Get the start date of the current sequence.
        var startDate = originalStartDate.advance(ee.Number(interval).multiply(i), units);

        // Get the end date of the current sequence.
        var endDate = originalStartDate.advance(
            ee.Number(interval).multiply(ee.Number(i).add(1)), units);

        return collection.filterDate(startDate, endDate).mosaic();
    }));
}