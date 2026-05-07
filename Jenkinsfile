pipeline {
    agent any

    environment {
        AWS_REGION   = "ap-south-1"
        ECR_REGISTRY = "857263388884.dkr.ecr.ap-south-1.amazonaws.com"
        ECR_URI      = "857263388884.dkr.ecr.ap-south-1.amazonaws.com/mpox-app"
        IMAGE_NAME   = "mpox-app"
    }

    stages {

        stage('Clone Code') {
            steps {
                checkout scm
            }
        }

        stage('Build Docker Image') {
            steps {
                sh 'docker build -t $IMAGE_NAME .'
            }
        }

        stage('Tag Image') {
            steps {
                sh 'docker tag $IMAGE_NAME:latest $ECR_URI:latest'
            }
        }

        stage('Login to ECR') {
            steps {
                withCredentials([[
                    $class: 'AmazonWebServicesCredentialsBinding',
                    credentialsId: 'aws-creds'
                ]]) {

                    sh '''
                        aws ecr get-login-password --region $AWS_REGION | \
                        docker login --username AWS --password-stdin $ECR_REGISTRY
                    '''
                }
            }
        }

        stage('Push Image to ECR') {
            steps {
                sh 'docker push $ECR_URI:latest'
            }
        }

        stage('Refresh ASG') {
            steps {
                withCredentials([[
                    $class: 'AmazonWebServicesCredentialsBinding',
                    credentialsId: 'aws-creds'
                ]]) {

                    sh '''
                        aws autoscaling start-instance-refresh \
                        --auto-scaling-group-name mpox-asg \
                        --region $AWS_REGION || true
                    '''
                }
            }
        }
    }
}