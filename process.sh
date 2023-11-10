#!/bin/bash
sort $1 | uniq | awk -F, '
$18 != -1 {
  total_rating += $18
  hotel_with_rating_count++
}

END {
  if (total_rating > 0) {
    printf("RATING_AVG %.2f\n", total_rating / hotel_with_rating_count)
  } else {
    print "RATINGA_AVG N/A\n"
  }
}'

sort $1 | uniq | awk -F, '
$7 != -1 {
  country = tolower($7)
  hotel_number[country]++
}

END {
  for (country in hotel_number) {
    print "HOTELNUMBER", country, hotel_number[country]
  }
}'

sort $1 | uniq | awk -F, '
$2 ~ /holiday inn/ {
  if ($12 > 0) {
    country = tolower($7)
    holiday_hotel_cleanliness[country] += $12
    holiday_hotel_count[country]++
  }
}

$2 ~ /hilton/ {
  if ($12 > 0) {
    country = tolower($7)
    hilton_hotel_cleanliness[country] += $12
    hilton_hotel_count[country]++
  }
}

END {
  for (country in holiday_hotel_cleanliness) {
    printf("CLEANLINESS %s %.2f %.2f\n",
      country,
      holiday_hotel_cleanliness[country] / holiday_hotel_count[country],
      hilton_hotel_cleanliness[country] / hilton_hotel_count[country])
  }
}
'

awk -F, '
$12 != -1, $18 != -1 {
  print $12, $18
}' $1 > /tmp/plot_file.csv
gnuplot << EOF
  set terminal png size 800,800
  set xlabel "overall ratingsource"
  set ylabel "cleanliness"
  set title "CLEANLINESS vs. OVERALL RATINGSOURCE"
  set output 'plot.png'
  set datafile separator " "
  set fit quiet logfile '/dev/null'
  f(x) = m*x + b
  fit f(x) '/tmp/plot_file.csv' using 2:1 via m, b
  plot '/tmp/plot_file.csv' using 2:1 with points title "", f(x) title "Linear regression"
EOF
