int rodaFrenteEsquerda1 = 13;
int rodaFrenteEsquerda2 = 12;

int rodaFrenteDireita1 = 11;
int rodaFrenteDireita2 = 10;

int rodaTrasEsquerda1 = 9;
int rodaTrasEsquerda2 = 8;

int rodaTrasDireita1 = 7;
int rodaTrasDireita2 = 6;

void rodaDireitaFrente() {
  digitalWrite(rodaFrenteDireita1, 255);
  digitalWrite(rodaFrenteDireita2, HIGH);
}

void rodaEsquerdaFrente() {
  digitalWrite(rodaFrenteEsquerda1, 255);
  digitalWrite(rodaFrenteEsquerda2, HIGH);
}

void rodaDireitaTras() {
  digitalWrite(rodaTrasDireita1, 255);
  digitalWrite(rodaTrasDireita2, HIGH);
}

void rodaEsquerdaTras() {
  digitalWrite(rodaTrasEsquerda1, 255);
  digitalWrite(rodaTrasEsquerda2, HIGH);
}

void setup()
{
  Serial.begin(9600);
  for (int i = 6; i < 14; i++) {
    pinMode(i, OUTPUT);
  }
}

void loop()
{
  while(Serial.available() == 0) {}
  String command = "";
  
  while(command != "stop") {
    command = Serial.readString();
    command.trim();

    if (command == "left") {
      Serial.println("Vira para ESQUERDA");
    } else if (command == "right") {
      Serial.println("Vira para DIREITA");
    } else if (command == "front") {
      Serial.println("Siga RETO");
    }
  }
}
