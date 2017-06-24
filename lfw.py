def read_pairs(pairs_filename):
  pairs = []
  with open(pairs_filename, 'r') as f:
  for line in f.readlines()[1:]:
      pair = line.strip().split()
      pairs.append(pair)
return np.array(pairs)

def get_paths(lfw_dir, pairs, file_ext):
  nrof_skiped_pairs = 0
  path_list = []
  issame_list = []
  for pair in pairs:
    if len(pair) == 3:
      path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])+'.'+file_ext)
      path1 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])+'.'+file_ext)
      issame = True
    elif len(pair) == 4:
      path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])+'.'+file_ext)
      path1 = os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])+'.'+file_ext)
      issame = False
    if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
      path_list += (path0,path1)
      issame_list.append(issame)
    else:
      nrof_skipped_pairs += 1
  if nrof_skipped_pairs>0:
    print('Skipped %d image pairs' % nrof_skipped_pairs)
  
  return path_list, issame_list

def evaluate(embeddings, actual_issame, nrof_folds=10):
  # Calculate evaluation metrics
  thresholds = np.arange(0, 4, 0.01)
  embeddings1 = embeddings[0::2]
  embeddings2 = embeddings[1::2]
  tpr, fpr, accuracy = facenet.calculate_roc(thresholds, embeddings1, embeddings2,
      np.asarray(actual_issame), nrof_folds=nrof_folds)
  thresholds = np.arange(0, 4, 0.001)
  val, val_std, far = facenet.calculate_val(thresholds, embeddings1, embeddings2,
      np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds)
  return tpr, fpr, accuracy, val, val_std, far