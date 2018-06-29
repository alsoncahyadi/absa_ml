// Chart.js scripts
// -- Set new default font family and font color to mimic Bootstrap's default styling
Chart.defaults.global.defaultFontFamily = '-apple-system,system-ui,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif';
Chart.defaults.global.defaultFontColor = '#292b2c';
// -- Bar Chart Example

function make_bar_chart() {
  var ctx = document.getElementById("myBarChart");
  return new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ["LALA", "February", "March", "April", "May", "June"],
      datasets: [
        {
          label: "Revenue+",
          backgroundColor: "rgba(2,117,216,1)",
          borderColor: "rgba(2,117,216,1)",
          data: [4215, 5312, 6251, 7841, 9821, 14984],
        },
        {
          label: "HAI",
          backgroundColor: "rgba(2,0,216,1)",
          borderColor: "rgba(2,0,216,1)",
          data: [4215, 5312, 6251, 7841, 9821, 14984],
        },
        {
          label: "LOLO",
          backgroundColor: "rgba(2,117,0,1)",
          borderColor: "rgba(2,117,0,1)",
          data: [4215, 5312, 6251, 7841, 9821, 14984],
        }
      ],
    },
    options: {
      scales: {
        xAxes: [{

        }
        //   {
        //   time: {
        //     unit: 'month'
        //   },
        //   gridLines: {
        //     display: false
        //   },
        //   ticks: {
        //     maxTicksLimit: 6
        //   }
        // }
        ],
        yAxes: [{
          stacked: true,
          ticks: {
            min: 0,
            max: 15000,
            maxTicksLimit: 5
          },
          gridLines: {
            display: true
          }
        }],
      },
      legend: {
        display: false
      }
    }
  });
}

var myLineChart = make_bar_chart();
// -- Pie Chart Example

function make_pie_chart() {
  console.log("MAKING PIE CHART")
  var ctx = document.getElementById("myPieChart");  
  return new Chart(ctx, {
    type: 'polarArea',
    data: {
      labels: ["YOHO", "Red", "Yellow", "Green"],
      datasets: [{
        data: [12.21, 15.58, 11.25, 8.32],
        backgroundColor: ['#007bff', '#dc3545', '#ffc107', '#28a745'],
      }],
    },
  });
}

var myPieChart = make_pie_chart();
