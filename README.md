Алгоритм для определения числа пересечений ребёр графа по его изображению. 
Использовалась следующая идея: с помощью детектора углов можно найти "углы графа" -- ими будут являться вершины и точки пересечения рёбер. Зная число вершин, можно найти число точек пересечения.
Из множества алгоритом детекторов углов был выбран SUSAN по совокупности факторов, в частности он хорошо отличает ребра от углов даже при "пикселизированности" ребер.
Области, полученные SUSAN, не отображали необходимую информацию, потому что находились не в вершинах/точках, а около них. Для этого все области, относящиеся к одной точке объединялись по принципу "лежит близко к области -- скорее всего, ей принадлежит". Недостаток проекта -- вычисление соответствующей функции занимает много времени на больших изображениях. 
Наконец, подсчитывалось число областей. Более простого алгоритма, чем найти чило объектов на изображении, который было бы возможно реализовать самостоятельно, увы, не придумалось.
