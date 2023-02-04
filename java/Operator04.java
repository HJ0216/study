public class Operator04 {

	public static void main(String[] args) {
		int num1 = 0, num2 = 0;
		boolean result;
		
		result = ((num1+=10) < 0 && (num2+=10) > 0);
		System.out.println("result: " + result);
		System.out.println("num1: " + num1 + ", num2: " + num2);
		// &&: (num1+=10) < 0 : False -> (num2+=10) > 0 수행 X
		
		result = ((num1+=10) > 0 || (num2+=10) > 0);
		System.out.println("result: " + result);
		System.out.println("num1: " + num1 + ", num2: " + num2);
		// ||: (num1+=10) > 0 : True -> (num2+=10) > 0 수행 X

	}
	
}


/*
&&: A && B -> A = False -> B 수행X
&: A & B -> A = False -> B 수행
||: A || B -> A = True -> B 수행X
|: A | B -> A = True -> B 수행

 */
