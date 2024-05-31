https://github.com/davidmoncas/meshroom_CLI


info :
"je ne veux pas je veux créer une application qui génère de l'objet 3D (.obj) ou autre format compatible à partir d'une seule photo 2d. 

Pour ça j'ai utiliser le modèle sam pour d'abord isoler l'objet par la segmentation et supprimer le fond l'image après ça j'ai utiliser le modèle de wonder3D pour générer les images vue de ['front', 'front_right', 'right', 'back', 'left', 'front_left'] et ensuite j'ai utiliser un de leurs algorithme qui est un modèle qui génère l'objet 3D en .obj, mais le truc c'est que avant d'avoir un bon résultat qui se rapporche un de l'objet en question il m'a fallu 500000 itération (epoch) , ce qui m'a pris plus de 5h avant de générer juste un chat en 3D, le processus est trop long et en plus ça demande énormement de ressource  gpu, donc je me suis dire que je vais explorer d'autre piste, c'est j'ai remarquer l'application polycam qui utilise la photogrametrie qui est appareent moins gourmant en terme de puissance. UNe fois que j'ai mes images génrer il va donc me rester la suite."
