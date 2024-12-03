rm(list=ls())


library(dplyr)
library(data.table)

dir()
# Leo el dataset con fread, que es eficiente
datos <- fread("competencia_03_crudo.csv")

# 1. Filtrar los datos para excluir los meses previos a enero 2020
dataset <- datos[foto_mes >= 202001]

# Calculamos el periodo consecutivo de cada cliente
dsimple <- dataset[, list(
  "pos" = .I,  # Guardamos la posición original para ordenar luego
  numero_de_cliente,
  periodo0 = foto_mes %/% 100 * 12 + foto_mes %% 100  # Conversión de foto_mes a periodo consecutivo
)]

# Ordenamos por cliente y periodo para asegurar la secuencia correcta
setorder(dsimple, numero_de_cliente, periodo0)

# Obtenemos los topes para los cálculos (el último periodo y el anteúltimo)
periodo_ultimo <- dsimple[, max(periodo0)]
periodo_anteultimo <- periodo_ultimo - 1

# Calculamos los leads (periodos siguientes) para cada cliente
dsimple[, c("periodo1", "periodo2") := shift(periodo0, n = 1:2, fill = NA, type = "lead"), by = numero_de_cliente]

# Creamos una columna para la clase ternaria
dsimple[, clase_ternaria := NA_character_]

# Calculamos BAJA+1 (el cliente no aparece en los siguientes dos meses)
dsimple[periodo0 < periodo_anteultimo &
          (is.na(periodo1) | periodo0 + 1 != periodo1) &
          (is.na(periodo2) | periodo0 + 2 != periodo2),
        clase_ternaria := "BAJA+1"]

# Calculamos BAJA+2 (el cliente aparece en el siguiente mes pero no en el mes dos meses después)
dsimple[periodo0 < periodo_anteultimo &
          (periodo0 + 1 == periodo1) &
          (is.na(periodo2) | periodo0 + 2 != periodo2),
        clase_ternaria := "BAJA+2"]

# Calculamos CONTINUA (el cliente aparece en los siguientes dos meses)
dsimple[periodo0 < periodo_anteultimo & (periodo0 + 2 == periodo2),
        clase_ternaria := "CONTINUA"]

# Verificamos el conteo de cada clase ternaria
print(dsimple[, .N, by = clase_ternaria])


# Finalmente, pegamos el resultado en el dataset original y lo ordenamos de nuevo por la posición original
setorder(dsimple, pos)
dataset[, clase_ternaria := dsimple$clase_ternaria]

table(as.factor(dataset$foto_mes), as.factor(dataset$clase_ternaria))

rm(datos, dsimple)


names(dataset)

# Calcular el porcentaje de ceros para cada columna
percent_zeros <- sapply(dataset, function(col) {
  mean(col == 0) * 100
})

# Ordenar las columnas por porcentaje de ceros en orden descendente
sorted_columns <- sort(percent_zeros, decreasing = TRUE)

# Seleccionar las primeras 30 columnas con mayor porcentaje de ceros
top_30_columns <- names(sorted_columns)[1:30]

# Eliminar estas columnas del dataset
dataset2 <- dataset[, (top_30_columns) := NULL]

rm(dataset)


names(dataset2)

write.csv(dataset2, "DatosComp3.csv", row.names = F)
