/**
 * minichart.js — tiny D3 v7 line chart for the cockpit pages.
 * Renders an SVG chart into a container div; theme-aware via an options color()
 * callback and CSS variables (--muted, --border) inherited from the page.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory(require("d3"));
  else root.MiniChart = factory(window.d3);
})(typeof self !== "undefined" ? self : this, function (d3) {
  "use strict";

  if (!d3) return function () { return { update: function () {} }; };

  function MiniChart(container, opts) {
    opts = opts || {};
    var height = opts.height || 148;
    var margin = { top: 26, right: 18, bottom: 24, left: 46 };
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
    var areaPath = g.append("path").attr("fill", "none");
    var path = g
      .append("path")
      .attr("fill", "none")
      .attr("stroke-width", 2.5)
      .attr("stroke-linejoin", "round")
      .attr("stroke-linecap", "round");
    var grid = g.append("g");
    var xAxis = g.append("g");
    var yAxis = g.append("g");

    if (opts.title) {
      svg
        .append("text")
        .attr("x", margin.left)
        .attr("y", 15)
        .style("font-size", "10.5px")
        .style("font-weight", 600)
        .style("fill", "var(--muted)")
        .text(opts.title);
    }

    function inner() {
      return {
        w: Math.max(80, (container.clientWidth || 600) - margin.left - margin.right),
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
      var ticks = Math.min(5, Math.max(2, Math.floor(box.h / 26)));
      grid
        .selectAll("line")
        .data(y.ticks(ticks))
        .join("line")
        .attr("x1", 0)
        .attr("x2", box.w)
        .attr("y1", y)
        .attr("y2", y)
        .style("stroke", "var(--border)")
        .style("stroke-dasharray", "3 4");

      // axes
      xAxis
        .attr("transform", "translate(0," + box.h + ")")
        .call(
          d3
            .axisBottom(x)
            .ticks(Math.max(2, Math.min(8, Math.floor(box.w / 130))))
            .tickSizeOuter(0)
            .tickSizeInner(-box.h),
        )
        .style("color", "var(--muted)")
        .style("font-size", "10px");
      yAxis
        .call(d3.axisLeft(y).ticks(ticks).tickSizeOuter(0))
        .style("color", "var(--muted)")
        .style("font-size", "10px");
      xAxis.selectAll("line").style("stroke", "var(--border)");
      yAxis.selectAll("line, path").style("stroke", "var(--border)");
      xAxis.selectAll("text").style("fill", "var(--muted)");
      yAxis.selectAll("text").style("fill", "var(--muted)");
      xAxis.selectAll(".domain").style("display", "none");

      if (data.length < 2) {
        path.attr("d", "");
        areaPath.attr("d", "");
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

      var color = opts.color ? opts.color() : "var(--primary)";
      var line = d3
        .line()
        .x(function (_, i) { return x(i); })
        .y(function (d) { return y(d); });
      path.datum(data).attr("d", line).style("stroke", color);

      // soft area fill under the curve (theme-matched gradient)
      var area = d3
        .area()
        .x(function (_, i) { return x(i); })
        .y0(box.h)
        .y1(function (d) { return y(d); });
      var gid = "grad-" + Math.floor(Math.random() * 1e6);
      var defs = svg.selectAll("defs").data([1]).join("defs");
      var grad = defs
        .selectAll("linearGradient")
        .data([gid])
        .join("linearGradient")
        .attr("id", gid)
        .attr("x1", 0)
        .attr("y1", 0)
        .attr("x2", 0)
        .attr("y2", 1);
      grad.selectAll("stop").data([0, 1]).join("stop")
        .attr("offset", function (d) { return d; })
        .attr("stop-color", color)
        .attr("stop-opacity", function (d) { return d === 0 ? 0.18 : 0; });
      areaPath.datum(data).attr("d", area).style("fill", "url(#" + gid + ")");
    }

    if (typeof ResizeObserver !== "undefined") {
      try {
        new ResizeObserver(function () { update(); }).observe(container);
      } catch (e) { /* ignore */ }
    }
    update();
    return { update: update, svg: svg };
  }

  return MiniChart;
});
