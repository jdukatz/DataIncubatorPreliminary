// main vis

function getClusterCounts(data)
{
    var clusterCounts = [{"id": 0, "count": 0, "cumulative": 0, "method": data[0].cluster_method},
                         {"id": 1, "count": 0, "cumulative": 0, "method": data[0].cluster_method},
                         {"id": 2, "count": 0, "cumulative": 0, "method": data[0].cluster_method},
                         {"id": 3, "count": 0, "cumulative": 0, "method": data[0].cluster_method}];
    for (var i=0; i<data.length; ++i) {
        var chyron = data[i];
        clusterId = chyron["cluster_id"];
        clusterCounts[clusterId].count++;
    }
    for (var i=0; i<clusterCounts.length; ++i) {
        if (i == 0) {
            clusterCounts[i].cumulative = 0
        } else {
            clusterCounts[i].cumulative = clusterCounts[i-1].count + clusterCounts[i-1].cumulative;
        }
        
    }
    return clusterCounts;
}

function plotBars(data, clusterNum)
{
    var barSvg = d3.select("#barsSvg");

    var totalPoints = data.length;
    var clusterCounts = getClusterCounts(data);

    var widthScale = d3.scaleLinear().domain([0, totalPoints]).range([0, 500]);
    var heightScale = d3.scaleQuantile().domain([1, 2, 3]).range([0, 100, 200]);
    var colorScale = d3.scaleQuantile().domain([0, 1, 2, 3]).range(["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]);

    barSvg.append("text")
        .attr("x", 550)
        .attr("y", heightScale(clusterNum) + 50)
        .attr("font-family", "sans-serif")
        .text(data[0].cluster_method)

    return barSvg.append("g")
        .attr("class", "rects")
        .selectAll("rect")
        .data(clusterCounts)
        .enter()
        .append("rect")
        .attr("id", function(d) { return d.id; })
        .attr("x", function(d) { return widthScale(d.cumulative); })
        .attr("y", function(d) { return heightScale(clusterNum); })
        .attr("width", function(d) { return widthScale(d.count); })
        .attr("height", 100)
        .attr("fill", function(d) { return colorScale(d.id); })
        .attr("stroke", "#000")
        .attr("stroke-width", "2px")
        .on("click", function() { 
            d3.selectAll("rect")
                .classed("hidden", function(d) { return d.method !== data[0].cluster_method; });
            return plotProjection(data);
        });
}

function plotProjection(data) 
{
    // remove displayed tooltips when we change plots
    d3.selectAll(".d3-tip.n").remove()

    // tooltip
    var tooltip = d3.tip()
        .attr("class", "d3-tip")
        .offset([-10,0])
        .html(function(d) {
            //debugger
            return d.channel + "<br><br>" + d.raw_text;
        });
    d3.select("#plotSvg").call(tooltip)

    currentPlot = data[0].cluster_method;
    var padding = 5;

    var colorScale = d3.scaleQuantile().domain([0, 1, 2, 3]).range(["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]);

    var maxX = d3.max(data, function(d) { return d["projected_vector"][0]; });
    var minX = d3.min(data, function(d) { return d["projected_vector"][0]; });
    var maxY = d3.max(data, function(d) { return d["projected_vector"][1]; });
    var minY = d3.min(data, function(d) { return d["projected_vector"][1]; });

    var xScale = d3.scaleLinear().domain([minX, maxX]).range([0 + padding, 500 - padding]);
    var yScale = d3.scaleLinear().domain([minY, maxY]).range([0 + padding, 500 - padding]);

    var circles = d3.select("#plotSvg").selectAll("circle");
    if (!circles.empty()) {
        circles.remove();
    }

    return d3.select("#plotSvg").append("g").attr("class", "dots")
        .selectAll("circle")
        .data(data) //data here will be all the data points
        .enter()
        .append("circle")
        .attr("class", "dots")
        .attr("r", 3)
        .attr("cx", function(d) { return xScale(d["projected_vector"][0]); })
        .attr("cy", function(d) { return yScale(d["projected_vector"][1]); })
        .attr("fill", function(d) { return colorScale(d["cluster_id"]); })
        .on("click", tooltip.show);
}

//define buttons
var projButtons = [
    {
        name: "tSNE",
        text: "t-SNE",
        click: function() {
            if (currentPlot == "km") {
                plotProjection(lv_tSNE_km);
            } else if (currentPlot == "agg") {
                plotProjection(lv_tSNE_agg);
            } else if (currentPlot == "spec") {
                plotProjection(lv_tSNE_spec);
            }
        }
    },
    {
        name: "PCA",
        text: "PCA",
        click: function() {
            if (currentPlot == "km") {
                plotProjection(lv_PCA_km);
            } else if (currentPlot == "agg") {
                plotProjection(lv_PCA_agg);
            } else if (currentPlot == "spec") {
                plotProjection(lv_PCA_spec);
            }
        }
    },
    {
        name: "MDS",
        text: "MDS",
        click: function() {
            if (currentPlot == "km") {
                plotProjection(lv_MDS_km);
            } else if (currentPlot == "agg") {
                plotProjection(lv_MDS_agg);
            } else if (currentPlot == "spec") {
                plotProjection(lv_MDS_spec);
            }
        }
    }];

var newsButtons = [
    {
        name: "FOX",
        text: "FOX",
        click: function() {
            d3.selectAll("circle")
                .classed("hidden", function(d) {
                    return d.channel !== "FOXNEWSW"
                });
        }
    },
    {
        name: "MSNBC",
        text: "MSNBC",
        click: function() {
            d3.selectAll("circle")
                .classed("hidden", function(d) {
                    return d.channel !== "MSNBCW"
                });
        }
    },
    {
        name: "CNN",
        text: "CNN",
        click: function() {
            d3.selectAll("circle")
                .classed("hidden", function(d) {
                    return d.channel !== "CNNW"
            });
        }
    },
    {
        name: "BBC",
        text: "BBC",
        click: function() {
            d3.selectAll("circle")
                .classed("hidden", function(d) {
                    return d.channel !== "BBCNEWS"
            });
        }
    }
];

// generate initial view
d3.select("#bars").append("svg")
    .attr("width", 600)
    .attr("height", 300)
    .attr("id", "barsSvg");

plotBars(lv_tSNE_km, 1);
plotBars(lv_tSNE_agg, 2);
plotBars(lv_tSNE_spec, 3);

//bind buttons
d3.select("#plot").append("g").attr("class", "buttons")
    .selectAll("button")
    .data(projButtons)
    .enter()
    .append("button")
    .attr("id", function(d) { return d.name; })
    .text(function(d) { return d.text; })
    .on("click", function(d) { return d.click(); });

d3.select("#plot").append("br")

d3.select("#plot").append("g").attr("class", "buttons")
    .selectAll("button")
    .data(newsButtons)
    .enter()
    .append("button")
    .attr("id", function(d) { return d.name; })
    .text(function(d) { return d.text; })
    .on("click", function(d) { return d.click(); });

d3.select("#plot").append("br")

var plotSvg = d3.select("#plot").append("svg")
    .attr("width", 500)
    .attr("height", 500)
    .attr("id", "plotSvg");

function keywordSearch() {
    d3.selectAll("circle")
        .classed("hidden", function(d) {
            searchTerm = document.getElementById("keywordSearch").value;
            return !d.raw_text.includes(searchTerm);
        });
}

d3.select("#search")
    .append("button")
    .attr("id", "searchButton")
    .text("Search")
    .on("click", keywordSearch);

plotProjection(lv_tSNE_km);


