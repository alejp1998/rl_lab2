/**
 * minichart.js — tiny D3 v7 line chart for the cockpit pages.
 * Renders an SVG chart into a container div; theme-aware via an options color()
 * callback and CSS variables (--muted, --border) inherited from the page.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports)
    module.exports = factory(require("d3"));
  else root.MiniChart = factory(window.d3);
})(typeof self !== "undefined" ? self : this, function (d3) {
  "use strict";

  if (!d3)
    return function () {
      return { update: function () {} };
    };

  function MiniChart(container, opts) {
    opts = opts || {};
    var height = opts.height || 148;
    var margin = { top: 24, right: 16, bottom: 22, left: 48 };
    var pad = opts.pad !== undefined ? opts.pad : 5;

    var svg = d3
      .select(container)
      .append("svg")
      .attr("width", "100%")
      .attr("height", height)
      .attr("role", "img")
      .style("display", "block");

    var g = svg
      .append("g")
      .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    var x = d3.scaleLinear();
    var y = d3.scaleLinear();
    var path = g
      .append("path")
      .attr("fill", "none")
      .attr("stroke-width", 2)
      .attr("stroke-linejoin", "round")
      .attr("stroke-linecap", "round");
    var grid = g.append("g");
    var xAxis = g.append("g");
    var yAxis = g.append("g");

    if (opts.title) {
      svg
        .append("text")
        .attr("x", margin.left + 2)
        .attr("y", 15)
        .style("font-size", "10px")
        .style("fill", "var(--muted)")
        .text(opts.title);
    }

    function inner() {
      return {
        w: Math.max(
          80,
          (container.clientWidth || 600) - margin.left - margin.right,
        ),
        h: height - margin.top - margin.bottom,
      };
    }

    function update() {
      var data = (opts.getData && opts.getData()) || [];
      var box = inner();
      x.range([0, box.w]).domain([0, Math.max(1, data.length - 1)]);
      var lo = data.length ? d3.min(data) - pad : 0;
      var hi = data.length ? d3.max(data) + pad : 1;
      if (!(hi > lo)) hi = lo + 1;
      y.range([box.h, 0]).domain([lo, hi]);

      // horizontal grid
      var ticks = Math.min(5, Math.max(2, Math.floor(box.h / 24)));
      grid
        .selectAll("line")
        .data(y.ticks(ticks))
        .join("line")
        .attr("x1", 0)
        .attr("x2", box.w)
        .attr("y1", y)
        .attr("y2", y)
        .style("stroke", "var(--border)")
        .style("stroke-dasharray", "3 3");

      // axes
      xAxis
        .attr("transform", "translate(0," + box.h + ")")
        .call(
          d3
            .axisBottom(x)
            .ticks(Math.max(2, Math.min(10, Math.floor(box.w / 90))))
            .tickSizeOuter(0),
        )
        .style("color", "var(--muted)")
        .style("font-size", "9px");
      yAxis
        .call(d3.axisLeft(y).ticks(ticks).tickSizeOuter(0))
        .style("color", "var(--muted)")
        .style("font-size", "9px");
      xAxis.selectAll("line").style("stroke", "var(--border)");
      yAxis.selectAll("line, path").style("stroke", "var(--border)");
      xAxis.selectAll("text, .tick text").style("fill", "var(--muted)");
      yAxis.selectAll("text, .tick text").style("fill", "var(--muted)");

      if (data.length < 2) {
        path.attr("d", "");
        g.selectAll(".empty")
          .data([1])
          .join("text")
          .attr("class", "empty")
          .attr("x", box.w / 2)
          .attr("y", box.h / 2)
          .style("text-anchor", "middle")
          .style("fill", "var(--muted)")
          .style("font-size", "12px")
          .style("font-weight", 600)
          .text(opts.emptyText || "training…");
        return;
      }
      g.selectAll(".empty").remove();
      var line = d3
        .line()
        .x(function (_, i) {
          return x(i);
        })
        .y(function (d) {
          return y(d);
        });
      path
        .datum(data)
        .attr("d", line)
        .style("stroke", opts.color ? opts.color() : "var(--primary)");
    }

    if (typeof ResizeObserver !== "undefined") {
      try {
        new ResizeObserver(function () {
          update();
        }).observe(container);
      } catch (e) {
        /* ignore */
      }
    }
    update();
    return { update: update, svg: svg };
  }

  return MiniChart;
});
