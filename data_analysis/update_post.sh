#filename must be the same as post name and in the files/ dir
export KNOWLEDGE_REPO="repos/${name_of_post}"
knowledge_repo init
knowledge_repo add -p ${name_of_post} files/${name_of_post}.ipynb --update
