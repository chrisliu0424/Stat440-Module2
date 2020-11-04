get_matrix = function(result,Ytest){
  # This function automately get prediction result from the corresponding column in the full prediciton dataset
  # Input: A 50000*p prediction matrix and the original Ytest dataset
  # Output: A 50000*2 prediction matrix
  # Required library: stringr
  prediction_df = Ytest
  value = strsplit(prediction_df$Id,":")
  prediction_df$row = as.numeric(str_extract(sapply(value,'[',1),'\\d+'))
  prediction_df$column = as.numeric(str_extract(sapply(value,'[',2),'\\d+'))
  for (i in 1:nrow(Ytest)) {
    prediction_df[i,'Value'] = result[i, prediction_df$column[i]]
  }
  prediction_df$Id = as.integer(prediction_df$row)
  prediction_df$column = NULL
  prediction_df$row = NULL
  prediction_df$Value = as.numeric(prediction_df$Value)
  return(prediction_df)
}