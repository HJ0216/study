public class Dice {

	public static void main(String[] args) {
		int dice1, dice2;

		dice1 = (int)(Math.random()*6+1);
		dice2 = (int)(Math.random()*6+1);

		String result = dice1 > dice2 ? "Dice1 win" : dice1 < dice2 ? "Dice2 win" : "Draw";
		System.out.println("Dice1: " + dice1 + ", Dice2: " + dice2 + ", Result: " + result);
		
//		dice1 = Math.random();
		// int <- Double : Type mismatch: cannot convert from double to int
		// Math.random(): static
		
	}
	
}