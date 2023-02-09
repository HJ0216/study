import java.util.*;

public class BaseBall_T {

	public static void main(String[] args) {
		Scanner scan = new Scanner(System.in);
		
		int[] com = new int[3];
		int[] user = new int[3];
		
		String yn;
		
		do {
			System.out.println("Are you ready to start Game(Y/N)? ");
			yn = scan.next();
			// yn은 내부에서 선언할 경우, 지역변수로 do{}에서만 사용 가능
		} while(!yn.equals("Y") && !yn.equals("y") && !yn.equals("N") && !yn.equals("n"));
		
		if(yn.equals("Y") || yn.equals("y")) {
			System.out.println("Start Game");
			
			
			// Computer Random
			for(int i=0; i<com.length; i++) {
				com[i] = (int)(Math.random()*9+1);
				
				 // 중복 제거
				for(int j=0; j<i; j++) {
					if (com[i]==com[j]) {
						i--;
						break;
					} // if
				} // for j
			} // for i
			
			System.out.println(com[0] + ", " + com[1] + ", " + com[2]);
			
			
			// 사용자 숫자 입력
			while(true) {
				
				int strike = 0; // strike, ball 초기화
				int ball = 0;
				
				System.out.println("\nEnter the number: ");
				int num = scan.nextInt();
				
				user[0] = num/100;
				user[1] = (num%100)/10;
				user[2] = (num%100)%10;

				/*
				String num = scan.next();
				
				user[0] = num.charAt(0);
				user[1] = num.charAt(1);
				user[2] = num.charAt(2);				
				 */
				
				System.out.println(user[0] + ", " + user[1] + ", " + user[2]);

				
				// comparison
				for(int i=0; i<com.length; i++) {
					for(int j=0; j<com.length; j++) {
						
						if(com[i]==user[j]) {
							if(i == j) {strike++;}
							else {ball++;}
						} // if
					} // for j
				} // for i
				
				System.out.println(strike + "스트라이크\t" + ball + "볼");
				
				if(strike==3) {
					System.out.println("Correct!");
					break; // break while(true)
				}
				
				
				
				/*
				com[0] == user[0], user[1], user[2]
				com[1] == user[0], user[1], user[2]
				com[2] == user[0], user[1], user[2]
				
				 */
			
			
			} // while
			
						
		} else System.out.println("Terminated Program");
		

		scan.close();
	}
	
}
