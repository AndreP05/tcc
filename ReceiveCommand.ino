int rodaFrenteEsquerda1 = 13;
int rodaFrenteEsquerda2 = 12;
int rodaFrenteEsquerdaSolo = 2;

int rodaFrenteDireita1 = 11;
int rodaFrenteDireita2 = 10;
int rodaFrenteDireitaSolo = 3;

int rodaTrasEsquerda1 = 9;
int rodaTrasEsquerda2 = 8;
int rodaTrasEsquerdaSolo = 4;

int rodaTrasDireita1 = 7;
int rodaTrasDireita2 = 6;
int rodaTrasDireitaSolo = 5;

void moverFrente() {
  digitalWrite(rodaFrenteEsquerdaSolo, LOW);
  digitalWrite(rodaFrenteEsquerda2, HIGH);
  digitalWrite(rodaFrenteDireitaSolo, LOW);
  digitalWrite(rodaFrenteDireita2, HIGH);
  
  digitalWrite(rodaTrasEsquerdaSolo, LOW);
  digitalWrite(rodaTrasEsquerda2, HIGH);
  digitalWrite(rodaTrasDireitaSolo, LOW);
  digitalWrite(rodaTrasDireita2, HIGH);
}

void moverTras() {
  digitalWrite(rodaFrenteEsquerdaSolo, HIGH);
  digitalWrite(rodaFrenteEsquerda2, LOW);
  digitalWrite(rodaFrenteDireitaSolo, HIGH);
  digitalWrite(rodaFrenteDireita2, LOW);
  
  digitalWrite(rodaTrasEsquerdaSolo, HIGH);
  digitalWrite(rodaTrasEsquerda2, LOW);
  digitalWrite(rodaTrasDireitaSolo, HIGH);
  digitalWrite(rodaTrasDireita2, LOW);
}

void virarDireita() {
  digitalWrite(rodaFrenteDireitaSolo, HIGH);
  digitalWrite(rodaFrenteDireita2, LOW);
  digitalWrite(rodaTrasDireitaSolo, HIGH);
  digitalWrite(rodaTrasDireita2, LOW);
  
  digitalWrite(rodaFrenteEsquerdaSolo, LOW);
  digitalWrite(rodaFrenteEsquerda2, HIGH);
  digitalWrite(rodaTrasEsquerdaSolo, LOW);
  digitalWrite(rodaTrasEsquerda2, HIGH);
}

void virarEsquerda() {
  digitalWrite(rodaFrenteEsquerdaSolo, HIGH);
  digitalWrite(rodaFrenteEsquerda2, LOW);
  digitalWrite(rodaTrasEsquerdaSolo, HIGH);
  digitalWrite(rodaTrasEsquerda2, LOW);
  
  digitalWrite(rodaFrenteDireitaSolo, LOW);
  digitalWrite(rodaFrenteDireita2, HIGH);
  digitalWrite(rodaTrasDireitaSolo, LOW);
  digitalWrite(rodaTrasDireita2, HIGH);
}

void setup()
{
  Serial.begin(9600);
  for (int i = 2; i < 14; i++) {
    pinMode(i, OUTPUT);
  }
  
  digitalWrite(rodaFrenteEsquerdaSolo, LOW);
  digitalWrite(rodaFrenteEsquerda2, LOW);
  digitalWrite(rodaFrenteDireitaSolo, LOW);
  digitalWrite(rodaFrenteDireita2, LOW);
  digitalWrite(rodaTrasEsquerdaSolo, LOW);
  digitalWrite(rodaTrasEsquerda2, LOW);
  digitalWrite(rodaTrasDireitaSolo, LOW);
  digitalWrite(rodaTrasDireita2, LOW);
}

void loop()
{
  digitalWrite(rodaFrenteEsquerda1, HIGH);
  digitalWrite(rodaFrenteDireita1, HIGH);
  digitalWrite(rodaTrasEsquerda1, HIGH);
  digitalWrite(rodaTrasDireita1, HIGH);
  
  while(Serial.available() == 0) {}
  String command = "";
  
  while (command != "stop") {
    command = Serial.readString();
    command.trim();
    
    int time = command.substring(4).toInt();
    command = command.substring(0, 4);

    if (command == "back") {
      moverTras();
    } else if (command == "righ") {
      virarDireita();
      delay(500);
      moverFrente();
    } else if (command == "left") {
      virarEsquerda();
      delay(500);
      moverFrente();
    }
    delay(time * 1000);
  }
}
