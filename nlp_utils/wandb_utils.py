import  wandb
import datetime as dt
def config_dict(model,batch_size):
    wandb_config = {'batch_size': batch_size }
    tf_config = model.get_config()

    layers = tf_config['layers']
    x=0
    for layer in layers:

        if layer['class_name']=='Conv2D':
            x = x + 1
            dict = {}
            dict['Type']=layer['class_name']
            dict['Kernel']=layer['config']['kernel_size'][0]
            dict['Stride']=layer['config']['strides'][0]
            dict['Padding']=layer['config']['padding']

            wandb_config['layer' + str(x)] = dict
        elif tf_config=='Dense':
            dict = {}
            dict['Type']=layer['class_name']
            dict['units']=layer['config']['units']
            wandb_config['layer'+str(x)]=dict
            x = x + 1

        return dict
def upload_best_model(project,job,descripcion,meta,path):
    '''

    :param projecto de W&B
    :param job: trabajo que hace la pipeline
    :param descripcion: una descripcion es formato str
    :param meta: Diccionario de datos
    :param path: path del modelo
    :return: Sube las bases a W&B
    '''
    with wandb.init(project=project, job_type=job) as run:
        # separate code for loading the datasets

        # Creamos el artifacto
        model_artifact = wandb.Artifact(
            "best model", type="model",
            description=str(dt.datetime.today())+'///'+descripcion,
            metadata=meta)
        model_artifact.add_file(path)
        wandb.save(path)

        # ✍ se sube el modelo a W&B
        run.log_artifact(model_artifact)
